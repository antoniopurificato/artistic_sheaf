import os
import json

INPUT_PATH = "Hertziana/triplets_hertziana_val.json"
OUTPUT_PATH = "Hertziana/triplets_hertziana_val.json"

# Optional: base directory where your images actually live
BASE_DIR = "Hertziana"   # change this if needed

# --- Load ---
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

original_len = len(data)

# --- Filter ---
cleaned_data = []
removed = 0

for entry in data:
    relative_path = entry.get("item1")

    full_path = os.path.join(BASE_DIR, relative_path)
    if os.path.exists(full_path):
        cleaned_data.append(entry)
    else:
        removed += 1

# --- Save ---
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print(f"Original entries: {original_len}")
print(f"Removed entries: {removed}")
print(f"Remaining entries: {len(cleaned_data)}")
print(f"Saved cleaned file to: {OUTPUT_PATH}")
