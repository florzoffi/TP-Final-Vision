from pathlib import Path
from typing import Optional
import shutil
import cv2
import numpy as np

def map_rgb_to_thermal(rgb_path: Path, t_split_root: Path) -> Optional[Path]:
    stem_rgb = rgb_path.stem      
    ext_rgb  = rgb_path.suffix    

    parts = stem_rgb.split("_")
    num_str = parts[-1]
    if not num_str.isdigit():
        return None

    num_rgb = int(num_str)
    num_t   = num_rgb - 1
    num_t_str = str(num_t).zfill(len(num_str))
    prefix = "_".join(parts[:-1])
    stem_t = f"{prefix}_{num_t_str}_R"
    cand = t_split_root / f"{stem_t}{ext_rgb}"
    if cand.exists():
        return cand

    candidates = list(t_split_root.glob(f"{stem_t}.*"))
    if len(candidates) == 0:
        return None
    return candidates[0]


def fuse_rgb_and_thermal(rgb_path: Path, t_path: Path) -> Optional[np.ndarray]:
    rgb = cv2.imread(str(rgb_path))  
    t   = cv2.imread(str(t_path), cv2.IMREAD_UNCHANGED)  

    if rgb is None or t is None:
        return None

    if t.ndim == 3:
        t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

    H, W = rgb.shape[:2]
    if t.shape[0] != H or t.shape[1] != W:
        t = cv2.resize(t, (W, H), interpolation=cv2.INTER_LINEAR)

    rgb_f = rgb.astype(np.float32) / 255.0
    t_f   = t.astype(np.float32)   / 255.0   

    fused = np.zeros_like(rgb_f)   
    fused[..., 0] = rgb_f[..., 2] 
    fused[..., 1] = rgb_f[..., 1] 
    fused[..., 2] = t_f           

    fused_u8 = (fused * 255.0).clip(0, 255).astype(np.uint8)
    return fused_u8

def build_rgbt_split(
    rgb_img_root: Path,
    t_img_root: Path,
    rgb_lbl_root: Path,
    rgbt_img_root: Path,
    rgbt_lbl_root: Path,
    split: str,
) -> None:
    rgb_dir = rgb_img_root / split
    t_dir   = t_img_root   / split

    out_img_dir = rgbt_img_root / split
    out_lbl_dir = rgbt_lbl_root / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    lbl_src_dir = rgb_lbl_root / split

    rgb_files = sorted(
        list(rgb_dir.glob("*.jpg")) +
        list(rgb_dir.glob("*.JPG")) +
        list(rgb_dir.glob("*.png")) +
        list(rgb_dir.glob("*.PNG"))
    )

    print(f"[{split}] Imágenes RGB encontradas: {len(rgb_files)}")

    created = 0

    for rgb_path in rgb_files:
        t_path = map_rgb_to_thermal(rgb_path, t_dir)
        if t_path is None:
            print(f"[WARN {split}] No se encontró térmica para {rgb_path.name}, se saltea.")
            continue

        fused = fuse_rgb_and_thermal(rgb_path, t_path)
        if fused is None:
            print(f"[WARN {split}] Error leyendo/fusionando {rgb_path.name} / {t_path.name}")
            continue

        out_img_path = out_img_dir / rgb_path.name
        cv2.imwrite(str(out_img_path), fused)

        lbl_name = rgb_path.stem + ".txt"
        lbl_src  = lbl_src_dir / lbl_name
        if lbl_src.exists():
            lbl_dst = out_lbl_dir / lbl_name
            shutil.copy2(lbl_src, lbl_dst)
        else:
            print(f"[WARN {split}] No se encontró label para {rgb_path.name}")

        created += 1
        print(f"[OK {split}] {rgb_path.name} -> {out_img_path.name} (T: {t_path.name})")

    print(f"[{split}] Total imágenes fusionadas creadas: {created}")
    yaml_content = """\
    path: data/format_rgbt

    train: images/train
    val: images/val
    test: images/test

    names:
        0: Cow
        1: Deer
        2: Horse
        """

    yaml_path = Path("data/format_rgbt/wildlife_rgbt.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Archivo creado: {yaml_path.resolve()}")