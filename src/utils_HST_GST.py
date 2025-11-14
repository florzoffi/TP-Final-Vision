# utils_multispectral.py

from pathlib import Path
from PIL import Image
import shutil
import re

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def build_hst(rgb_img: Image.Image, t_img: Image.Image) -> Image.Image:
    rgb_img = rgb_img.convert("RGB")
    t_img = t_img.convert("L")

    if t_img.size != rgb_img.size:
        t_img = t_img.resize(rgb_img.size, Image.BILINEAR)

    hsv_img = rgb_img.convert("HSV")
    h, s, v = hsv_img.split()

    hst_img = Image.merge("RGB", (h, s, t_img))
    return hst_img


def build_gst(rgb_img: Image.Image, t_img: Image.Image) -> Image.Image:

    rgb_img = rgb_img.convert("RGB")
    t_img = t_img.convert("L")

    if t_img.size != rgb_img.size:
        t_img = t_img.resize(rgb_img.size, Image.BILINEAR)

    g = rgb_img.convert("L")
    hsv_img = rgb_img.convert("HSV")
    h, s, v = hsv_img.split()

    gst_img = Image.merge("RGB", (g, s, t_img))
    return gst_img


def find_thermal_for_rgb_in_split(rgb_img_path: Path, t_img_dir: Path) -> Path | None:

    stem = rgb_img_path.stem  
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None

    num_str = m.group(1)
    num = int(num_str)
    prefix = stem[: m.start(1)]
    width = len(num_str)

    candidate_stems = [
        f"{prefix}{num-1:0{width}d}_R",  
        f"{prefix}{num:0{width}d}_R",   
    ]

    for cand_stem in candidate_stems:
        for ext in IMG_EXTS:
            cand = t_img_dir / f"{cand_stem}{ext}"
            if cand.exists():
                return cand

    return None


def create_hst_gst_from_roots(
    root_rgb: str | Path,
    root_t: str | Path,
    out_hst_root: str | Path,
    out_gst_root: str | Path,
):

    root_rgb = Path(root_rgb)
    root_t = Path(root_t)
    out_hst_root = Path(out_hst_root)
    out_gst_root = Path(out_gst_root)

    splits = ["train", "val", "test"]

    created_hst = 0
    created_gst = 0
    skipped_no_t = 0
    skipped_no_lbl = 0

    for split in splits:
        rgb_img_dir = root_rgb / "images" / split
        rgb_lbl_dir = root_rgb / "labels" / split
        t_img_dir = root_t / "images" / split

        print(f"\n Split {split}")
        print(f"RGB img dir: {rgb_img_dir} (exists: {rgb_img_dir.exists()})")
        print(f"T img dir:   {t_img_dir}   (exists: {t_img_dir.exists()})")

        hst_img_dir = out_hst_root / "images" / split
        gst_img_dir = out_gst_root / "images" / split
        hst_lbl_dir = out_hst_root / "labels" / split
        gst_lbl_dir = out_gst_root / "labels" / split

        hst_img_dir.mkdir(parents=True, exist_ok=True)
        gst_img_dir.mkdir(parents=True, exist_ok=True)
        hst_lbl_dir.mkdir(parents=True, exist_ok=True)
        gst_lbl_dir.mkdir(parents=True, exist_ok=True)

        n_rgb_split = 0

        for rgb_path in rgb_img_dir.rglob("*"):
            if rgb_path.suffix not in IMG_EXTS:
                continue
            n_rgb_split += 1

            stem = rgb_path.stem
            lbl_path = rgb_lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                skipped_no_lbl += 1
                continue

            t_path = find_thermal_for_rgb_in_split(rgb_path, t_img_dir)
            if t_path is None:
                skipped_no_t += 1
                continue

            rgb_img = Image.open(rgb_path)
            t_img = Image.open(t_path)

            hst_img = build_hst(rgb_img, t_img)
            gst_img = build_gst(rgb_img, t_img)

            out_hst_img = hst_img_dir / f"{stem}.png"
            out_gst_img = gst_img_dir / f"{stem}.png"
            hst_img.save(out_hst_img)
            gst_img.save(out_gst_img)

            shutil.copy2(lbl_path, hst_lbl_dir / f"{stem}.txt")
            shutil.copy2(lbl_path, gst_lbl_dir / f"{stem}.txt")

            created_hst += 1
            created_gst += 1

        print(f"  RGB images in this split: {n_rgb_split}")

    print("\nHST/GST datasets created")
    print(f"- Images HST created: {created_hst}")
    print(f"- Images GST created: {created_gst}")
    print(f"- Skipped bc of t missing: {skipped_no_t}")
    print(f"- Skipped bc of rgb missing: {skipped_no_lbl}")