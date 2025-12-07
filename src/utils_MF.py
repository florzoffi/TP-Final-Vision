from pathlib import Path
import torch
import cv2
import re

def results_to_tensor(res):
    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    xyxy = res.boxes.xyxy  
    conf = res.boxes.conf.view(-1, 1)  
    cls  = res.boxes.cls.view(-1, 1)   

    return torch.cat([xyxy, conf, cls], dim=1).detach().cpu()

def box_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(x2 - x1, 0.0)
    inter_h = max(y2 - y1, 0.0)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union

def yolo_middle_fusion(
    res_rgb,
    res_t,
    iou_match: float = 0.5,
    conf_penalty_single: float = 0.9,
):
    boxes_rgb = results_to_tensor(res_rgb)  
    boxes_t   = results_to_tensor(res_t)    

    if boxes_rgb.numel() == 0 and boxes_t.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32)
    fused_boxes = []
    used_t = set()

    for i in range(boxes_rgb.shape[0]):
        b_r = boxes_rgb[i]
        br_xyxy = b_r[:4]
        br_conf = b_r[4].item()
        br_cls  = int(b_r[5].item())

        best_j = -1
        best_iou = 0.0

        for j in range(boxes_t.shape[0]):
            if j in used_t:
                continue

            b_t = boxes_t[j]
            bt_xyxy = b_t[:4]
            bt_conf = b_t[4].item()
            bt_cls  = int(b_t[5].item())

            if bt_cls != br_cls:
                continue

            iou = box_iou_xyxy(br_xyxy, bt_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_match:
            b_t = boxes_t[best_j]
            bt_xyxy = b_t[:4]
            bt_conf = b_t[4].item()
            bt_cls  = int(b_t[5].item())

            w_r = br_conf
            w_t = bt_conf
            w_sum = max(w_r + w_t, 1e-6)

            xyxy_fused = (br_xyxy * w_r + bt_xyxy * w_t) / w_sum
            conf_fused = (br_conf + bt_conf) / 2.0
            cls_fused  = br_cls

            fused_boxes.append(
                torch.tensor(
                    [xyxy_fused[0], xyxy_fused[1], xyxy_fused[2], xyxy_fused[3], conf_fused, cls_fused],
                    dtype=torch.float32
                )
            )
            used_t.add(best_j)
        else:
            b_new = b_r.clone()
            b_new[4] = b_new[4] * conf_penalty_single
            fused_boxes.append(b_new)

    for j in range(boxes_t.shape[0]):
        if j not in used_t:
            b_t = boxes_t[j].clone()
            b_t[4] = b_t[4] * conf_penalty_single
            fused_boxes.append(b_t)

    if len(fused_boxes) == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    fused_tensor = torch.stack(fused_boxes, dim=0)


    return fused_tensor

def draw_fused_boxes(img_bgr, fused_tensor, class_names=None):
    img = img_bgr.copy()
    if fused_tensor.numel() == 0:
        return img

    fused = fused_tensor.cpu().numpy()
    for x1, y1, x2, y2, conf, cls in fused:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        label = str(cls)
        if class_names is not None:
            if isinstance(class_names, dict):
                label = class_names.get(cls, str(cls))
            elif isinstance(class_names, (list, tuple)):
                if 0 <= cls < len(class_names):
                    label = class_names[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
    return img

def print_metrics(name, metrics):
    print(f"{name} @ IoU 0.5")
    print("-" * (len(name) + 12))
    print("mAP:      ", metrics["mAP"])
    print("Precision:", metrics["Precision"])
    print("Recall:   ", metrics["Recall"])
    print("F1:       ", metrics["F1"])
    print("AP por clase:", metrics["AP_per_class"])


def run_middle_fusion_split(
    model_rgb,
    model_t,
    class_names,
    rgb_dir: Path,
    t_dir: Path,
    out_img_dir: Path,
    out_pred_dir: Path,
    img_size: int = 640,
):
    rgb_paths = sorted(
        list(rgb_dir.glob("*.jpg")) +
        list(rgb_dir.glob("*.JPG")) +
        list(rgb_dir.glob("*.png")) +
        list(rgb_dir.glob("*.PNG"))
    )

    print(f"Encontradas {len(rgb_paths)} imágenes RGB en {rgb_dir}.")

    for img_rgb_path in rgb_paths:
        img_name = img_rgb_path.name
        stem_rgb = img_rgb_path.stem
        ext_rgb  = img_rgb_path.suffix

        m = re.search(r"(.*_DJI_)(\d{4})$", stem_rgb)
        if m:
            prefix = m.group(1)           
            num    = int(m.group(2))     

            cand_stems = [
                f"{prefix}{num:04d}_R",       
                f"{prefix}{num-1:04d}_R",     
                f"{prefix}{num+1:04d}_R",     
            ]

            img_t_path = None
            for cs in cand_stems:
                p = t_dir / f"{cs}{ext_rgb}"
                if p.exists():
                    img_t_path = p
                    break

                matches = list(t_dir.glob(f"{cs}.*"))
                if matches:
                    img_t_path = matches[0]
                    break

            if img_t_path is None:
                print(f"[WARN] No se encontró térmica para {img_name}")
                print("       probé:", ", ".join(f"{cs}.*" for cs in cand_stems))
                continue

        else:
            candidates = list(t_dir.glob(f"{stem_rgb}*R.*"))
            if not candidates:
                print(f"[WARN] No se encontró térmica para {img_name} (fallback)")
                continue
            img_t_path = candidates[0]

        res_rgb = model_rgb(str(img_rgb_path), imgsz=img_size, device="cpu", verbose=False)[0]
        res_t   = model_t(str(img_t_path),    imgsz=img_size, device="cpu", verbose=False)[0]

        fused = yolo_middle_fusion(
            res_rgb,
            res_t,
            iou_match=0.5,
            conf_penalty_single=0.9,
        )

        pred_txt_path = out_pred_dir / f"{stem_rgb}.txt"

        if fused.numel() == 0:
            open(pred_txt_path, "w").close()
        else:
            img_bgr = cv2.imread(str(img_rgb_path))
            H, W = img_bgr.shape[:2]

            fused_np = fused.cpu().numpy()
            with open(pred_txt_path, "w") as f:
                for x1, y1, x2, y2, conf, cls in fused_np:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w  = (x2 - x1)
                    h  = (y2 - y1)

                    cx /= W
                    cy /= H
                    w  /= W
                    h  /= H

                    line = f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}\n"
                    f.write(line)
        img_bgr = cv2.imread(str(img_rgb_path))
        img_out = draw_fused_boxes(img_bgr, fused, class_names)
        out_img_path = out_img_dir / img_name
        cv2.imwrite(str(out_img_path), img_out)

        print(f"[OK] Middle Fusion: {img_name} -> img:{out_img_path.name}, preds:{pred_txt_path.name}")

    print(f"Listo: imágenes en {out_img_dir} y predicciones en {out_pred_dir}")
