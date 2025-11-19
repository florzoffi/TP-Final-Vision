import torch
import cv2

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




