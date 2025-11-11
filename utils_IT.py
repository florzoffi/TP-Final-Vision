from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def find_root(ROOTS, CLASSES):
    for r in ROOTS:
        for p in r.rglob('*'):
            if all((p/c).is_dir() for c in CLASSES): return p
            
        
def draw_yolo_boxes(img_path, CLASSES):
    txt_path = img_path.with_suffix('.txt')
    if not txt_path.exists():
        print(f"⚠️ No hay anotación para {img_path.name}")
        return

    img = Image.open(img_path)
    w, h = img.size
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    with open(txt_path, 'r') as f:
        for line in f:
            cls_id, x_c, y_c, bw, bh = map(float, line.strip().split())
            x = (x_c - bw / 2) * w
            y = (y_c - bh / 2) * h
            bw *= w
            bh *= h

            rect = patches.Rectangle((x, y), bw, bh, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            label = CLASSES[int(cls_id)] if int(cls_id) < len(CLASSES) else f"cls {int(cls_id)}"
            ax.text(x, y - 5, label, color='lime', fontsize=10, weight='bold')

    plt.axis('off')
    plt.show()