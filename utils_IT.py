from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
        
        
def draw_yolo_boxes(img_path, CLASSES):
    txt_path = img_path.with_suffix('.txt')
    if not txt_path.exists():
        print(f"No hay anotación para {img_path.name}")
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

    
def load_images_by_class(data_dir='data', classes=('Horse', 'Deer', 'Cow')):
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    DATASET_ROOT = Path(data_dir)
    images = {}
    for cls in classes:
        class_dir = DATASET_ROOT / cls.lower()
        images[cls] = [p for p in class_dir.rglob('*') if p.suffix in IMG_EXTS]

    for c, lst in images.items():
        print(f"{c}: {len(lst)} imágenes")
        for p in lst[:2]:
            draw_yolo_boxes(p, classes)
    
    return images