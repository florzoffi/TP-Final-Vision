import torch
import cv2
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def box_iou_xyxy(box1, box2):
    """
    box1: [4], box2: [4] en formato [x1, y1, x2, y2]
    devuelve IoU escalar
    """
    # intersección
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_w = max(ix2 - ix1, 0.0)
    inter_h = max(iy2 - iy1, 0.0)
    inter = inter_w * inter_h

    # áreas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


# utils_LF.py

# --------------------------------------------------
# IoU para formato YOLO (cx, cy, w, h normalizado)
# --------------------------------------------------
def iou_xywh(box1, box2):
    """
    IoU entre dos cajas en formato [cx, cy, w, h] (normalizado 0-1)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min = x1 - w1/2
    y1_min = y1 - h1/2
    x1_max = x1 + w1/2
    y1_max = y1 + h1/2

    x2_min = x2 - w2/2
    y2_min = y2 - h2/2
    x2_max = x2 + w2/2
    y2_max = y2 + h2/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area + 1e-9

    return inter_area / union


# --------------------------------------------------
# Evaluación: AP por clase + mAP + métricas globales
# --------------------------------------------------
def evaluate_yolo_predictions(pred_dir, gt_dir, num_classes, iou_threshold=0.5):
    """
    pred_dir: carpeta con .txt pred (class cx cy w h conf)
    gt_dir:   carpeta con .txt GT (class cx cy w h)
    """
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)

    all_scores = {c: [] for c in range(num_classes)}
    all_tp     = {c: [] for c in range(num_classes)}
    n_gt       = {c: 0  for c in range(num_classes)}

    gt_files = sorted(gt_dir.glob("*.txt"))

    for gt_file in gt_files:
        stem = gt_file.stem
        pred_file = pred_dir / f"{stem}.txt"

        # GT
        if gt_file.exists() and gt_file.stat().st_size > 0:
            gts = np.loadtxt(gt_file, ndmin=2)
        else:
            gts = np.zeros((0, 5))

        for gt in gts:
            cls_g = int(gt[0])
            if cls_g in n_gt:
                n_gt[cls_g] += 1

        # Predicciones
        if pred_file.exists() and pred_file.stat().st_size > 0:
            preds = np.loadtxt(pred_file, ndmin=2)
        else:
            preds = np.zeros((0, 6))

        if preds.shape[0] == 0:
            continue

        for c in range(num_classes):
            gts_c   = gts[gts[:, 0] == c] if gts.shape[0] > 0 else np.zeros((0, 5))
            preds_c = preds[preds[:, 0] == c] if preds.shape[0] > 0 else np.zeros((0, 6))

            if preds_c.shape[0] == 0:
                continue

            used = np.zeros(len(gts_c), dtype=bool)

            for pr in preds_c:
                _, cx, cy, w, h, conf = pr
                box_p = np.array([cx, cy, w, h])

                best_iou = 0
                best_j = -1
                for j, gt in enumerate(gts_c):
                    _, cxg, cyg, wg, hg = gt
                    box_g = np.array([cxg, cyg, wg, hg])
                    iou = iou_xywh(box_p, box_g)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= iou_threshold and best_j >= 0 and not used[best_j]:
                    all_tp[c].append(1)
                    all_scores[c].append(conf)
                    used[best_j] = True
                else:
                    all_tp[c].append(0)
                    all_scores[c].append(conf)

    ap_per_class = []
    precision_global_tp = 0
    precision_global_fp = 0
    recall_global_fn    = 0

    for c in range(num_classes):
        scores = np.array(all_scores[c])
        tps    = np.array(all_tp[c])
        if n_gt[c] == 0:
            ap_per_class.append(np.nan)
            continue
        if len(scores) == 0:
            ap_per_class.append(0.0)
            continue

        order = np.argsort(-scores)
        tps = tps[order]
        fps = 1 - tps

        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)

        recalls    = tp_cum / (n_gt[c] + 1e-9)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        # AP = área bajo curva PR
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        ap_per_class.append(ap)

        precision_global_tp += tp_cum[-1]
        precision_global_fp += fp_cum[-1]
        recall_global_fn    += n_gt[c] - tp_cum[-1]

    valid_aps = [ap for ap in ap_per_class if not np.isnan(ap)]
    mAP = float(np.mean(valid_aps)) if len(valid_aps) > 0 else 0.0

    TP = precision_global_tp
    FP = precision_global_fp
    FN = recall_global_fn

    precision_global = TP / (TP + FP + 1e-9)
    recall_global    = TP / (TP + FN + 1e-9)
    f1_global        = 2 * precision_global * recall_global / (precision_global + recall_global + 1e-9)

    return {
        "AP_per_class": ap_per_class,
        "mAP": mAP,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision_global,
        "Recall": recall_global,
        "F1": f1_global
    }


# --------------------------------------------------
# Matriz de confusión
# --------------------------------------------------
def confusion_matrix_yolo(pred_dir, gt_dir, num_classes, iou_threshold=0.5):
    """
    Devuelve una matriz (num_classes+1)x(num_classes+1)
    Última fila/columna: background (no detectado / falsa alarma)
    """
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)

    cm = np.zeros((num_classes+1, num_classes+1), dtype=int)

    gt_files = sorted(gt_dir.glob("*.txt"))

    for gt_file in gt_files:
        stem = gt_file.stem
        pred_file = pred_dir / f"{stem}.txt"

        if gt_file.exists() and gt_file.stat().st_size > 0:
            gts = np.loadtxt(gt_file, ndmin=2)
        else:
            gts = np.zeros((0, 5))

        if pred_file.exists() and pred_file.stat().st_size > 0:
            preds = np.loadtxt(pred_file, ndmin=2)
        else:
            preds = np.zeros((0, 6))

        used_gt = np.zeros(len(gts), dtype=bool)

        for pr in preds:
            cls_p = int(pr[0])
            box_p = pr[1:5]

            best_iou = 0
            best_j = -1
            best_cls_g = None

            for j, gt in enumerate(gts):
                cls_g = int(gt[0])
                box_g = gt[1:5]
                iou = iou_xywh(box_p, box_g)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
                    best_cls_g = cls_g

            if best_iou >= iou_threshold and best_j >= 0 and not used_gt[best_j]:
                if best_cls_g < num_classes and cls_p < num_classes:
                    cm[best_cls_g, cls_p] += 1
                used_gt[best_j] = True
            else:
                if cls_p < num_classes:
                    cm[num_classes, cls_p] += 1

        for j, gt in enumerate(gts):
            if not used_gt[j]:
                cls_g = int(gt[0])
                if cls_g < num_classes:
                    cm[cls_g, num_classes] += 1

    return cm


# --------------------------------------------------
# Plots (confusion matrix + AP por clase)
# --------------------------------------------------
def plot_confusion_matrix(cm, class_labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="w" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    plt.show()


def plot_ap_bar(ap_per_class, class_labels, title="AP por clase"):
    fig, ax = plt.subplots()
    vals = [ap if not np.isnan(ap) else 0.0 for ap in ap_per_class]
    ax.bar(class_labels, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP @ IoU 0.5")
    ax.set_title(title)
    plt.show()


def yolo_late_fusion(pred_rgb, pred_t,
                     iou_det=0.6,
                     prob_thr=0.45,
                     iou_match=0.3):
    """
    pred_rgb, pred_t: objetos Results[0] de Ultralytics (una imagen)
    devuelve lista de detecciones fusionadas: [x1, y1, x2, y2, conf, cls]
    """
    boxes_r = pred_rgb.boxes.xyxy.cpu().numpy()
    conf_r  = pred_rgb.boxes.conf.cpu().numpy()
    cls_r   = pred_rgb.boxes.cls.cpu().numpy()

    boxes_t = pred_t.boxes.xyxy.cpu().numpy()
    conf_t  = pred_t.boxes.conf.cpu().numpy()
    cls_t   = pred_t.boxes.cls.cpu().numpy()

    used_t = set()
    fused_dets = []

    # 1) intentar emparejar cajas RGB con T de la misma clase
    for i in range(len(boxes_r)):
        best_j = -1
        best_iou = 0.0
        for j in range(len(boxes_t)):
            if j in used_t:
                continue
            if cls_r[i] != cls_t[j]:
                continue
            iou = box_iou_xyxy(boxes_r[i], boxes_t[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_match:
            # hay match RGB–T
            cr, ct = conf_r[i], conf_t[best_j]
            # regla PMS del paper
            if best_iou >= iou_det or max(cr, ct) >= prob_thr:
                # usamos la caja del modelo más confiado
                if cr >= ct:
                    box = boxes_r[i]
                    c   = cr
                    cls = cls_r[i]
                else:
                    box = boxes_t[best_j]
                    c   = ct
                    cls = cls_t[best_j]
                fused_dets.append([*box, c, cls])
            used_t.add(best_j)

        else:
            # RGB sin pareja -> se acepta sólo si la confianza es alta
            if conf_r[i] >= prob_thr:
                fused_dets.append([*boxes_r[i], conf_r[i], cls_r[i]])

    # 2) cajas T que quedaron sin usar
    for j in range(len(boxes_t)):
        if j in used_t:
            continue
        if conf_t[j] >= prob_thr:
            fused_dets.append([*boxes_t[j], conf_t[j], cls_t[j]])

    if len(fused_dets) == 0:
        return torch.empty((0, 6))

    return torch.tensor(fused_dets)

def draw_fused_boxes(img_bgr, fused_tensor, class_names=None):
    """
    img_bgr: imagen en BGR (cv2.imread)
    fused_tensor: tensor Nx6 -> [x1, y1, x2, y2, conf, cls]
    class_names: lista/dict con nombres de clases (opcional)
    """
    img = img_bgr.copy()
    if fused_tensor.numel() == 0:
        return img

    fused = fused_tensor.cpu().numpy()
    for x1, y1, x2, y2, conf, cls in fused:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        label = f"{cls}"
        if class_names is not None:
            label = class_names.get(cls, str(cls))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA)
    return img





