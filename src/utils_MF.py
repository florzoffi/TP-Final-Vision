from pathlib import Path
import torch
import cv2
import numpy as np
from utils_LF import plot_ap_bar


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
    """
    Aplica Middle Fusion sobre todas las imágenes RGB de un directorio,
    matcheando con sus térmicas, y guarda:
      - imágenes con cajas en out_img_dir
      - predicciones .txt (formato YOLO) en out_pred_dir
    """
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

        # ---------- mapear a térmica: num-1 + "_R" ----------
        parts = stem_rgb.split("_")
        num_str = parts[-1]
        if not num_str.isdigit():
            print(f"[WARN] {img_name}: no puedo leer número, salto.")
            continue

        num_rgb = int(num_str)
        num_t   = num_rgb - 1
        num_t_str = str(num_t).zfill(len(num_str))
        prefix = "_".join(parts[:-1])
        stem_t = f"{prefix}_{num_t_str}_R"

        img_t_path = t_dir / f"{stem_t}{ext_rgb}"
        if not img_t_path.exists():
            candidates = list(t_dir.glob(f"{stem_t}.*"))
            if len(candidates) == 0:
                print(f"[WARN] No se encontró térmica para {img_name} (esperaba {stem_t}{ext_rgb})")
                continue
            img_t_path = candidates[0]

        # ---------- inferencias ----------
        res_rgb = model_rgb(str(img_rgb_path), imgsz=img_size, device="cpu", verbose=False)[0]
        res_t   = model_t(str(img_t_path),    imgsz=img_size, device="cpu", verbose=False)[0]

        # ---------- Middle Fusion ----------
        fused = yolo_middle_fusion(
            res_rgb,
            res_t,
            iou_match=0.5,
            conf_penalty_single=0.9,
        )

        # ---------- guardar predicciones en TXT (formato YOLO) ----------
        pred_txt_path = out_pred_dir / f"{stem_rgb}.txt"

        if fused.numel() == 0:
            # archivo vacío si no hay detecciones
            open(pred_txt_path, "w").close()
        else:
            # leemos la imagen para conocer H,W (para normalizar)
            img_bgr = cv2.imread(str(img_rgb_path))
            H, W = img_bgr.shape[:2]

            fused_np = fused.cpu().numpy()
            with open(pred_txt_path, "w") as f:
                for x1, y1, x2, y2, conf, cls in fused_np:
                    # convertir a cx,cy,w,h normalizados
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

        # ---------- dibujar y guardar imagen ----------
        img_bgr = cv2.imread(str(img_rgb_path))
        img_out = draw_fused_boxes(img_bgr, fused, class_names)
        out_img_path = out_img_dir / img_name
        cv2.imwrite(str(out_img_path), img_out)

        print(f"[OK] Middle Fusion: {img_name} -> img:{out_img_path.name}, preds:{pred_txt_path.name}")

    print(f"✅ Listo: imágenes en {out_img_dir} y predicciones en {out_pred_dir}")
