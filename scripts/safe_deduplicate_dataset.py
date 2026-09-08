import os
import sys
import shutil
import json
import cv2
import hashlib
from pathlib import Path
from collections import defaultdict

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")

def run_safe_deduplication():
    img_dir = Path("dataset_dusit/images")
    meta_path = Path("dataset_dusit/metadata.json")
    backup_dir = Path("dataset_dusit/backup_duplicates")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. Backup metadata.json
    meta_bak = Path("dataset_dusit/metadata.json.bak")
    if meta_path.exists():
        shutil.copy2(meta_path, meta_bak)
        print(f"📦 Backup created: {meta_bak}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"Initial metadata entries: {len(meta)}")

    # 2. Index all image files and compute 16x16 perceptual dHash
    files = sorted([f for f in img_dir.iterdir() if f.is_file()])
    print(f"Total image files in folder: {len(files)}")

    hashes = defaultdict(list)
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, (17, 16))
        diff = resized[:, 1:] > resized[:, :-1]
        h = hashlib.md5(diff.tobytes()).hexdigest()
        hashes[h].append(f)

    # 3. For each group with >1 images, verify with 100% exact pixel diff
    files_to_remove = set()
    dedup_log = []

    for h, group in hashes.items():
        if len(group) <= 1:
            continue

        # Sort group: prioritize keeping img_xxxx over act_xxxx, then lower filename
        def sort_priority(path: Path):
            name = path.name
            # If name starts with img_, priority 0 (keep), else act_ priority 1
            is_img = 0 if name.startswith("img_") else 1
            return (is_img, name)

        sorted_group = sorted(group, key=sort_priority)
        primary_file = sorted_group[0]
        primary_img = cv2.imread(str(primary_file))

        for candidate in sorted_group[1:]:
            cand_img = cv2.imread(str(candidate))
            if primary_img.shape == cand_img.shape:
                pixel_diff = float(cv2.absdiff(primary_img, cand_img).mean())
                # STRICT: Must be 100% exact identical pixels
                if pixel_diff == 0.0:
                    files_to_remove.add(candidate.name)
                    dedup_log.append({
                        "kept_file": primary_file.name,
                        "duplicate_file": candidate.name,
                        "resolution": f"{primary_img.shape[1]}x{primary_img.shape[0]}",
                        "pixel_diff": pixel_diff
                    })

    print(f"\n🔍 Strict Verification Result:")
    print(f"  - Verified 100% Exact Pixel Duplicates: {len(files_to_remove)} files")

    # 4. Safely MOVE duplicates to backup folder
    moved_count = 0
    for filename in sorted(files_to_remove):
        src_path = img_dir / filename
        dst_path = backup_dir / filename
        if src_path.exists():
            shutil.move(src_path, dst_path)
            moved_count += 1

    print(f"✅ Safely moved {moved_count} duplicate files to: {backup_dir.resolve()}")

    # 5. Clean metadata.json
    clean_meta = [m for m in meta if m.get("filename") not in files_to_remove]
    # Re-index IDs sequentially
    for idx, m in enumerate(clean_meta, 1):
        m["id"] = idx
    meta_path.write_text(json.dumps(clean_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📄 Updated {meta_path} — Remaining clean entries: {len(clean_meta)}")

    # 6. Save audit report
    report_file = Path("dataset_dusit/dedup_report.json")
    report_file.write_text(json.dumps(dedup_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📋 Audit report saved: {report_file}")

if __name__ == "__main__":
    run_safe_deduplication()
