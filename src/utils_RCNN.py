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
