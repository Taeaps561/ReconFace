import os
import shutil
import json
import sys
from pathlib import Path

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
dst_dir = ROOT_DIR / "dataset_dusit" / "images"
meta_file = ROOT_DIR / "dataset_dusit" / "metadata.json"

print("==================================================================")
print("📦 MERGING VITCHA DATASET (ADDING ALL ~3,000 IMAGES TO DATASET)")
print("==================================================================")

existing_meta = []
if meta_file.exists():
    existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))

print(f"Current images in dataset_dusit: {len(existing_meta)}")

existing_filenames = {m["filename"] for m in existing_meta}
existing_sizes = {m.get("size_bytes", 0) for m in existing_meta}

next_id = max((m.get("id", 0) for m in existing_meta), default=0) + 1
new_entries = []
copied_count = 0

for y in sorted(src_base.iterdir()):
    if not y.is_dir():
        continue
    year = y.name
    for f in sorted(y.iterdir()):
        if not f.is_file() or f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        sz = f.stat().st_size
        if sz <= 1000:
            continue
        
        # Check if already present by size and filename
        dest_filename = f.name
        if dest_filename in existing_filenames or sz in existing_sizes:
            continue

        dest_path = dst_dir / dest_filename
        try:
            shutil.copy2(f, dest_path)
            copied_count += 1
            existing_filenames.add(dest_filename)
            existing_sizes.add(sz)

            new_entries.append({
                "id": next_id,
                "filename": dest_filename,
                "file_path": str(dest_path.resolve()),
                "page_title": f"Suan Dusit Photo ({year}) - {f.stem}",
                "page_url": f"Year {year}",
                "size_bytes": sz,
            })
            next_id += 1

            if copied_count % 200 == 0:
                print(f" -> Copied {copied_count} new images to {dst_dir}...", flush=True)
        except Exception as e:
            print(f"Error copying {f.name}: {e}")

print(f"✅ Successfully copied {copied_count} new unique images!")

combined_meta = existing_meta + new_entries
meta_file.write_text(json.dumps(combined_meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"📄 Updated {meta_file} with {len(combined_meta)} total images!")
print("==================================================================")
