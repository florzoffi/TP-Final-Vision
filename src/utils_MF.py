from pathlib import Path
import torch
import cv2
import numpy as np


# ---------------------------------------------------------
# 1. Helper: extraer cajas de un objeto Results de Ultralytics
#    Devuelve tensor Nx6: [x1,y1,x2,y2,conf,cls]
# ---------------------------------------------------------
def results_to_tensor(res):
    """
    res: ultralytics.engine.results.Results
    return: torch.Tensor [N,6] en CPU (xyxy, conf, cls)
    """
    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    xyxy = res.boxes.xyxy  # [N,4]
    conf = res.boxes.conf.view(-1, 1)  # [N,1]
    cls  = res.boxes.cls.view(-1, 1)   # [N,1]

    return torch.cat([xyxy, conf, cls], dim=1).detach().cpu()


# ---------------------------------------------------------
# 2. IoU en xyxy (para matchear cajas entre RGB y T)
# ---------------------------------------------------------
def box_iou_xyxy(box1, box2):
    """
    box1, box2: tensores [4] con [x1,y1,x2,y2]
    devuelve escalar IoU
    """
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


# ---------------------------------------------------------
# 3. Middle Fusion tipo "Weighted Box Fusion"
#    - Matchea cajas RGB y T por IoU + clase
#    - Funde coords y conf como promedio ponderado
#    - Las no matcheadas se conservan (tal vez penalizadas)
# ---------------------------------------------------------
def yolo_middle_fusion(
    res_rgb,
    res_t,
    iou_match: float = 0.5,
    conf_penalty_single: float = 0.9,
):
    """
    Middle Fusion simple entre detecciones RGB y T.

    - res_rgb, res_t: Results de Ultralytics
    - iou_match: IoU mínimo para considerar que dos cajas (misma clase) representan el mismo objeto
    - conf_penalty_single: factor para bajar levemente la conf de cajas que vienen solo de una rama

    Return:
        torch.Tensor [M,6] -> [x1,y1,x2,y2,conf,cls] fusionadas
    """
    boxes_rgb = results_to_tensor(res_rgb)  # [Nr,6]
    boxes_t   = results_to_tensor(res_t)    # [Nt,6]

    if boxes_rgb.numel() == 0 and boxes_t.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    fused_boxes = []
    used_t = set()

    # 1) Fusionar cajas que matchean (misma clase + IoU alto)
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
            # fuse RGB + T (weighted by confidence)
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
            # caja solo de RGB, la conservamos pero penalizamos conf
            b_new = b_r.clone()
            b_new[4] = b_new[4] * conf_penalty_single
            fused_boxes.append(b_new)

    # 2) Agregar cajas de T que no fueron usadas
    for j in range(boxes_t.shape[0]):
        if j not in used_t:
            b_t = boxes_t[j].clone()
            b_t[4] = b_t[4] * conf_penalty_single
            fused_boxes.append(b_t)

    if len(fused_boxes) == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    fused_tensor = torch.stack(fused_boxes, dim=0)

    # 3) opcional: NMS ligero propio (para limpiar solapamientos excesivos)
    #    por simplicidad no lo hacemos acá; se podría agregar si hace falta.

    return fused_tensor


# ---------------------------------------------------------
# 4. Dibujar cajas fusionadas (podés usar la misma firma que en Late-F)
# ---------------------------------------------------------
def draw_fused_boxes(img_bgr, fused_tensor, class_names=None):
    """
    img_bgr: imagen BGR (cv2.imread)
    fused_tensor: tensor [N,6] con [x1,y1,x2,y2,conf,cls]
    class_names: dict o lista {id: "nombre"}
    """
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
