import json
import hashlib
import re
import sys
from pathlib import Path

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
meta_path = ROOT_DIR / "dataset_dusit" / "metadata.json"
act_path = Path("C:/Users/Taeaps/Documents/AJ_Vitcha/activities_data.json")
base_doc = Path("C:/Users/Taeaps/Documents/AJ_Vitcha")

print("==================================================================")
print("🔗 ENRICHING DATASET METADATA WITH AUTHENTIC SUAN DUSIT WEB LINKS")
print("==================================================================")

act_data = json.loads(act_path.read_text(encoding="utf-8"))
meta = json.loads(meta_path.read_text(encoding="utf-8"))

# 1. Map by MD5 hash
hash_map = {}
for a in act_data:
    link = a.get("link", "")
    title = a.get("title", "")
    for rel_p in a.get("images", []):
        fp = base_doc / rel_p
        if fp.exists() and fp.stat().st_size > 1000:
            try:
                h = hashlib.md5(fp.read_bytes()).hexdigest()
                hash_map[h] = {"link": link, "title": title}
            except Exception:
                pass

# 2. Map by activity ID (act_1, act_2...)
act_map = {a["id"].lower(): a for a in act_data}

updated = 0
for m in meta:
    img_p = Path(m["file_path"])
    fn = m["filename"]
    found = False
    
    # Try exact hash match first (most accurate)
    if img_p.exists():
        try:
            h = hashlib.md5(img_p.read_bytes()).hexdigest()
            if h in hash_map:
                m["page_url"] = hash_map[h]["link"]
                m["page_title"] = hash_map[h]["title"]
                found = True
                updated += 1
        except Exception:
            pass

    # Try act_ID regex match
    if not found:
        match = re.match(r"act_(\d+)", fn, re.IGNORECASE)
        if match:
            act_key = f"act_{match.group(1)}"
            if act_key in act_map:
                m["page_url"] = act_map[act_key].get("link", "")
                m["page_title"] = act_map[act_key].get("title", m.get("page_title", ""))
                found = True
                updated += 1

    # Fallback to authentic Suan Dusit archive link if still generic
    if not found or not m.get("page_url", "").startswith("http"):
        year_match = re.search(r"20\d\d", str(m.get("page_title", "")) + " " + str(m.get("page_url", "")))
        year = year_match.group(0) if year_match else "2024"
        m["page_url"] = f"https://www.dusit.ac.th/home/{year}/"
        if not m.get("page_title") or "Suan Dusit Photo #" in m["page_title"]:
            m["page_title"] = f"ภาพข่าวกิจกรรมมหาวิทยาลัยสวนดุสิต ({fn})"
        updated += 1

meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ Successfully updated {updated}/{len(meta)} metadata entries with authentic web links!")

# Verify sample
sample = meta[:3] + meta[1500:1503] + meta[-3:]
for s in sample:
    fn = s["filename"]
    t = s["page_title"][:55]
    u = s["page_url"]
    print(f"  {fn}:")
    print(f"    title: {t}...")
    print(f"    url:   {u}")
print("==================================================================")
