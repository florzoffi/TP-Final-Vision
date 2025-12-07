import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import re

class FusionBlock(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_t, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels_rgb + in_channels_t,
                                 out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv3x3 = nn.Conv2d(out_channels,
                                 out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se_fc1 = nn.Linear(out_channels, out_channels // 16)
        self.se_fc2 = nn.Linear(out_channels // 16, out_channels)

    def se(self, x):
        b, c, h, w = x.shape
        y = x.mean(dim=(2, 3))        
        y = F.relu(self.se_fc1(y))    
        y = torch.sigmoid(self.se_fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y

    def forward(self, f_rgb, f_t):
        x = torch.cat([f_rgb, f_t], dim=1)   
        x = F.relu(self.bn1(self.conv1x1(x)))
        x = F.relu(self.bn2(self.conv3x3(x)))
        x = self.se(x)   
        return x


class WildlifeRGBTDataset(Dataset):
    """
    Dataset para Aerial Wildlife Image Repository (Cow/Deer/Horse)
    con imágenes RGB y térmicas alineadas por nombre de archivo.
    """
    def __init__(self, rgb_img_root, t_img_root, lbl_root, split, img_size=640):
        self.rgb_dir  = rgb_img_root / split
        self.t_dir    = t_img_root   / split
        self.lbl_dir  = lbl_root     / split
        self.img_size = img_size

        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        self.rgb_imgs = sorted(
            [p for p in self.rgb_dir.iterdir() if p.suffix.lower() in exts]
        )
        self.ids = [p.stem for p in self.rgb_imgs]

        print(f"[{split}] imágenes encontradas:", len(self.ids))

    def load_image(self, path):
        img = Image.open(path).convert("RGB")         
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img).astype(np.float32) / 255.0 
        arr = np.transpose(arr, (2, 0, 1))             
        return torch.from_numpy(arr)                   

    def load_labels(self, img_id):
        lbl_path = self.lbl_dir / f"{img_id}.txt"
        if not lbl_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        targets = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = map(float, parts)
                targets.append([cls, cx, cy, w, h])

        if len(targets) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        return torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # RGB
        rgb_path = self.rgb_dir / f"{img_id}.jpg"
        if not rgb_path.exists():
            found = list(self.rgb_dir.glob(f"{img_id}.*"))
            assert len(found) > 0, f"No se encontró RGB para {img_id}"
            rgb_path = found[0]

        # Térmica 
        t_path = self.t_dir / rgb_path.name
        if not t_path.exists():
            found_t = list(self.t_dir.glob(f"{img_id}.*"))
            assert len(found_t) > 0, f"No se encontró T para {img_id}"
            t_path = found_t[0]

        img_rgb = self.load_image(rgb_path)  
        img_t   = self.load_image(t_path)    
        labels  = self.load_labels(img_id)   
        return img_rgb, img_t, labels, img_id


def collate_fn(batch):
    imgs_rgb = []
    imgs_t   = []
    targets  = []
    ids      = []

    for i, (im_rgb, im_t, lbls, img_id) in enumerate(batch):
        imgs_rgb.append(im_rgb)
        imgs_t.append(im_t)
        if lbls.numel() > 0:
            bi = torch.full((lbls.shape[0], 1), i, dtype=torch.float32)
            targets.append(torch.cat([bi, lbls], dim=1))
        ids.append(img_id)

    imgs_rgb = torch.stack(imgs_rgb, dim=0)
    imgs_t   = torch.stack(imgs_t,   dim=0)

    if len(targets) == 0:
        targets_cat = torch.zeros((0, 6), dtype=torch.float32)
    else:
        targets_cat = torch.cat(targets, dim=0)

    return imgs_rgb, imgs_t, targets_cat, ids

IMG_SIZE_DEFAULT = 640  

class PatchFusionNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) 
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) 
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) 
        x = x.view(x.size(0), -1)                      
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))                 
        return x


def extract_rgbt_patch(rgb_path, t_path, box, patch_size=64, img_size=640):
    img_rgb = cv2.imread(str(rgb_path))                            
    img_t   = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)        
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))
    img_t   = cv2.resize(img_t,   (img_size, img_size))

    x1, y1, x2, y2 = box
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_size - 1, int(x2))
    y2 = min(img_size - 1, int(y2))

    if x2 <= x1 or y2 <= y1:
        patch_rgb = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        patch_t   = np.zeros((patch_size, patch_size),     dtype=np.uint8)
    else:
        patch_rgb = img_rgb[y1:y2, x1:x2, :]
        patch_t   = img_t[y1:y2, x1:x2]

        patch_rgb = cv2.resize(patch_rgb, (patch_size, patch_size))
        patch_t   = cv2.resize(patch_t,   (patch_size, patch_size))

    patch_rgb = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2RGB)

    patch_rgb = patch_rgb.astype(np.float32) / 255.0
    patch_t   = patch_t.astype(np.float32) / 255.0

    patch_t = np.expand_dims(patch_t, axis=-1)
    patch_rgbt = np.concatenate([patch_rgb, patch_t], axis=-1)  
    patch_rgbt = np.transpose(patch_rgbt, (2, 0, 1))            

    return torch.from_numpy(patch_rgbt.astype(np.float32))


def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter   = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    union = area1 + area2 - inter + 1e-6
    return inter / union


class PatchFusionDataset(Dataset):
    def __init__(self, rgb_dir, t_dir, preds_dir, gt_dir,
                 img_size=640, patch_size=64,
                 iou_pos_th=0.5, iou_neg_th=0.3,
                 max_samples_per_img=50):
        super().__init__()
        self.rgb_dir   = Path(rgb_dir)
        self.t_dir     = Path(t_dir)
        self.preds_dir = Path(preds_dir)
        self.gt_dir    = Path(gt_dir)
        self.img_size  = img_size
        self.patch_size = patch_size
        self.iou_pos_th = iou_pos_th
        self.iou_neg_th = iou_neg_th
        self.max_samples_per_img = max_samples_per_img

        self.ids = [p.stem for p in sorted(self.preds_dir.glob("*.txt"))]
        self.samples = []
        self._build_samples()

    def _load_boxes_xyxy_from_yolo(self, path: Path):
        boxes = []

        if not path.exists():
            return boxes

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                cx, cy, w, h = map(float, parts[1:5])

                W = self.img_size
                H = self.img_size

                x1 = (cx - w / 2.0) * W
                y1 = (cy - h / 2.0) * H
                x2 = (cx + w / 2.0) * W
                y2 = (cy + h / 2.0) * H

                boxes.append([x1, y1, x2, y2])

        return boxes


    def _build_samples(self):
        for img_id in self.ids:
            gt_path   = self.gt_dir    / f"{img_id}.txt"
            pred_path = self.preds_dir / f"{img_id}.txt"

            gt_boxes   = self._load_boxes_xyxy_from_yolo(gt_path)
            pred_boxes = self._load_boxes_xyxy_from_yolo(pred_path)

            if len(pred_boxes) == 0:
                continue

            local = []
            for box in pred_boxes:
                best_iou = 0.0
                for gt in gt_boxes:
                    best_iou = max(best_iou, iou_xyxy(box, gt))

                if best_iou >= self.iou_pos_th:
                    label = 1.0
                    local.append((img_id, box, label))
                elif best_iou <= self.iou_neg_th:
                    label = 0.0
                    local.append((img_id, box, label))

            if len(local) > self.max_samples_per_img:
                local = local[:self.max_samples_per_img]

            self.samples.extend(local)

        print(f"PatchFusionDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, box, label = self.samples[idx]

        # --- RGB ---
        rgb_path = self.rgb_dir / f"{img_id}.jpg"
        if not rgb_path.exists():
            found_rgb = list(self.rgb_dir.glob(f"{img_id}.*"))
            if len(found_rgb) == 0:
                raise FileNotFoundError(f"No se encontró imagen RGB para {img_id}")
            rgb_path = found_rgb[0]

        # --- Térmica ---
        t_path = self.t_dir / f"{img_id}.jpg"
        if not t_path.exists():
            found_t = list(self.t_dir.glob(f"{img_id}.*"))
            if len(found_t) == 0:
                t_path = rgb_path
            else:
                t_path = found_t[0]

        patch = extract_rgbt_patch(
            rgb_path,
            t_path,
            box,
            patch_size=self.patch_size,
            img_size=self.img_size,
        )

        y = torch.tensor([label], dtype=torch.float32)
        return patch, y

def find_thermal_for_id(t_dir: Path, img_id: str):
    t_dir = Path(t_dir)
    cands = list(t_dir.glob(f"{img_id}.*"))
    if cands:
        return cands[0]

    cands = list(t_dir.glob(f"{img_id}_R.*"))
    if cands:
        return cands[0]

    parts = img_id.split("_")
    last = parts[-1]
    if last.isdigit():
        prefix = "_".join(parts[:-1])
        n = int(last)
        for k in [n - 1, n + 1]:
            if k < 0:
                continue
            num_str = str(k).zfill(len(last))
            cands = list(t_dir.glob(f"{prefix}_{num_str}.*"))
            if cands:
                return cands[0]
            cands = list(t_dir.glob(f"{prefix}_{num_str}_R.*"))
            if cands:
                return cands[0]
    return None

def refine_with_patch_fusion(
    patch_model,
    rgb_dir,
    t_dir,
    preds_late_dir,
    preds_pf_dir,
    img_size=640,
    patch_size=64,
    score_thr=0.5,
    device="cpu",
):
    patch_model.eval()
    patch_model.to(device)

    rgb_dir        = Path(rgb_dir)
    t_dir          = Path(t_dir)
    preds_late_dir = Path(preds_late_dir)
    preds_pf_dir   = Path(preds_pf_dir)

    preds_pf_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(preds_late_dir.glob("*.txt"))
    print(f"[PatchFusion] Refinando {len(txt_files)} imágenes...")

    for txt_path in txt_files:
        img_id = txt_path.stem  
        rgb_path = None
        for ext in [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]:
            cand = rgb_dir / f"{img_id}{ext}"
            if cand.exists():
                rgb_path = cand
                break
        if rgb_path is None:
            found_rgb = list(rgb_dir.glob(f"{img_id}.*"))
            if len(found_rgb) == 0:
                print(f"[WARN] No se encontró RGB para {img_id}, salto esta imagen.")
                continue
            rgb_path = found_rgb[0]
        t_path = None

        m = re.search(r"(.*_DJI_)(\d{4})$", img_id)
        if m:
            prefix = m.group(1)
            num    = int(m.group(2))    

            cand_stems = [
                f"{prefix}{num-1:04d}_R",
                f"{prefix}{num:04d}_R",
                f"{prefix}{num+1:04d}_R",
            ]

            for cs in cand_stems:
                matches = list(t_dir.glob(f"{cs}.*"))
                if matches:
                    t_path = matches[0]
                    break

        if t_path is None:
            matches = list(t_dir.glob(f"{img_id}*R.*"))
            if matches:
                t_path = matches[0]

        if t_path is None:
            print(f"[WARN] No se encontró térmica para {img_id}, salto esta imagen.")
            continue

        with open(txt_path, "r") as f:
            raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if len(raw_lines) == 0:
            (preds_pf_dir / txt_path.name).write_text("")
            continue

        kept_lines = []

        for line in raw_lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            vals = list(map(float, parts))
            cls = vals[0]
            cx, cy, w, h = vals[1:5]
            orig_conf = vals[5] if len(vals) > 5 else 1.0

            x1 = (cx - w / 2.0) * img_size
            y1 = (cy - h / 2.0) * img_size
            x2 = (cx + w / 2.0) * img_size
            y2 = (cy + h / 2.0) * img_size
            box = (x1, y1, x2, y2)

            patch = extract_rgbt_patch(
                rgb_path,
                t_path,
                box,
                patch_size=patch_size,
                img_size=img_size,
            )  
            patch = patch.unsqueeze(0).to(device)  
            with torch.no_grad():
                scores = patch_model(patch)
                score = scores.view(-1)[0].item()

            if score < score_thr:
                continue
            new_conf = float(orig_conf * score)
            kept_lines.append(
                f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {new_conf:.4f}\n"
            )

        out_txt = preds_pf_dir / txt_path.name
        if len(kept_lines) == 0:
            out_txt.write_text("")
        else:
            with open(out_txt, "w") as f_out:
                f_out.writelines(kept_lines)

    print(f"[PatchFusion] Listo. Predicciones refinadas en {preds_pf_dir}")
