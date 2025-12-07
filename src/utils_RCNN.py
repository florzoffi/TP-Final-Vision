from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class WildlifeYOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        # admitimos jpg/png por las dudas
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        self.images = sorted([p for p in self.images_dir.rglob("*") if p.suffix in exts])
        self.transforms = transforms

        # 3 clases + background (que es 0 en Faster R-CNN)
        self.class_names = ["Cow", "Deer", "Horse"]  # ajustá si tu orden es distinto

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # archivo de labels YOLO
        label_path = self.labels_dir / (img_path.stem + ".txt")

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts)
                    cls = int(cls)

                    # YOLO (cx,cy,w,h) normalizado → px
                    cx *= w
                    cy *= h
                    bw *= w
                    bh *= h

                    xmin = cx - bw / 2
                    ymin = cy - bh / 2
                    xmax = cx + bw / 2
                    ymax = cy + bh / 2

                    boxes.append([xmin, ymin, xmax, ymax])
                    # Faster R-CNN usa 0 como background, así que nuestras clases arrancan en 1
                    labels.append(cls + 1)

        if len(boxes) == 0:
            # necesario para que no explote; rara vez debería pasar
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([], dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

def get_transform(train):
    transforms = [T.ToTensor()]
    # si querés, acá después podés agregar flips, color jitter, etc.
    return T.Compose(transforms)

def evaluate_fasterrcnn(model, data_loader, device, iou_thresholds=IOU_THRESHOLDS, num_classes=3):
    model.eval()
    all_metrics = {}
    global_tp = 0
    global_fp = 0
    global_fn = 0
    per_class_data = {
        thr: {
            cls_id: {
                "scores": [],
                "tp_flags": [],
                "num_gt": 0,
            }
            for cls_id in range(1, num_classes + 1)
        }
        for thr in iou_thresholds
    }

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].cpu().numpy()
                gt_labels = tgt["labels"].cpu().numpy()

                pred_boxes = out["boxes"].cpu().numpy()
                pred_scores = out["scores"].cpu().numpy()
                pred_labels = out["labels"].cpu().numpy()

                order = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]
                pred_labels = pred_labels[order]

                for thr in iou_thresholds:
                    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

                    for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
                        if pl == 0:  
                            continue

                        best_iou = 0.0
                        best_gt_idx = -1
                        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                            if gl != pl:
                                continue
                            iou = box_iou(pb, gb)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j

                        is_tp = False
                        if best_iou >= thr and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                            is_tp = True
                            gt_matched[best_gt_idx] = True

                        pc = per_class_data[thr][int(pl)]
                        pc["scores"].append(ps)
                        pc["tp_flags"].append(1 if is_tp else 0)

                    for gl in gt_labels:
                        if gl == 0:
                            continue
                        per_class_data[thr][int(gl)]["num_gt"] += 1

                thr = 0.5
                gt_matched_global = np.zeros(len(gt_boxes), dtype=bool)

                for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
                    if pl == 0:
                        continue
                    best_iou = 0.0
                    best_gt_idx = -1
                    for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if gl != pl:
                            continue
                        iou = box_iou(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                    if best_iou >= thr and best_gt_idx >= 0 and not gt_matched_global[best_gt_idx]:
                        global_tp += 1
                        gt_matched_global[best_gt_idx] = True
                    else:
                        global_fp += 1

                global_fn += int((~gt_matched_global).sum())

    mAP_50 = 0.0
    mAP_50_90 = 0.0
    valid_classes_50 = 0
    valid_classes_5090 = 0

    ap_per_class_iou = {thr: {} for thr in iou_thresholds}

    for thr in iou_thresholds:
        for cls_id in range(1, num_classes + 1):
            data = per_class_data[thr][cls_id]
            scores = np.array(data["scores"])
            tp_flags = np.array(data["tp_flags"])
            num_gt = data["num_gt"]

            if num_gt == 0:
                ap = np.nan
            else:
                order = np.argsort(-scores)
                scores = scores[order]
                tp_flags = tp_flags[order]

                fp_flags = 1 - tp_flags

                cum_tp = np.cumsum(tp_flags)
                cum_fp = np.cumsum(fp_flags)

                precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
                recall = cum_tp / max(num_gt, 1)

                ap = compute_ap(precision, recall)

            ap_per_class_iou[thr][CLASS_NAMES[cls_id]] = ap

        valid_aps = [v for v in ap_per_class_iou[thr].values() if not np.isnan(v)]
        mAP_thr = float(np.mean(valid_aps)) if len(valid_aps) > 0 else float("nan")

        if abs(thr - 0.5) < 1e-6:
            mAP_50 = mAP_thr
            valid_classes_50 = len(valid_aps)

        if len(valid_aps) > 0:
            mAP_50_90 += mAP_thr
            valid_classes_5090 += 1

    mAP_50_90 = mAP_50_90 / max(len(iou_thresholds), 1)

    precision_global = global_tp / max(global_tp + global_fp, 1)
    recall_global = global_tp / max(global_tp + global_fn, 1)
    f1_global = 0.0
    if precision_global + recall_global > 0:
        f1_global = 2 * precision_global * recall_global / (precision_global + recall_global)

    accuracy_global = global_tp / max(global_tp + global_fp + global_fn, 1)

    metrics = {
        "mAP@0.5": mAP_50,
        "mAP@0.5:0.9": mAP_50_90,
        "AP_per_class": ap_per_class_iou,   
        "precision": precision_global,
        "recall": recall_global,
        "f1": f1_global,
        "accuracy": accuracy_global,
    }
    return metrics

def box_iou(box1, box2):
    """
    box1: [4]  (x1, y1, x2, y2)
    box2: [4]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_ap(precision, recall):
    """
    AP como área bajo la curva P-R (regla del trapecio).
    precision, recall: arrays ordenados por recall creciente.
    """
    idx = np.argsort(recall)
    recall = recall[idx]
    precision = precision[idx]
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2.0
    return ap