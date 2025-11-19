from pathlib import Path
import cv2
import numpy as np
from utils_MF import results_to_tensor   

def save_results_to_txt(res, img_path, out_dir):
    """
    res: Results de YOLO
    img_path: Path de la imagen usada
    out_dir: carpeta donde guardar el .txt
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(img_path).stem
    txt_path = out_dir / f"{stem}.txt"

    boxes = results_to_tensor(res)  # [N,6] xyxy, conf, cls

    if boxes.numel() == 0:
        open(txt_path, "w").close()
        return

    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    boxes_np = boxes.cpu().numpy()
    with open(txt_path, "w") as f:
        for x1, y1, x2, y2, conf, cls in boxes_np:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = (x2 - x1)
            h  = (y2 - y1)

            cx /= W; cy /= H; w /= W; h /= H

            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")
