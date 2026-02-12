import json
import re
from collections import Counter

# ---- Canonicalization rules ----
MEDIUM_ALIASES = {
    "oil": "Oil",
    "tempera": "Tempera",
    "fresco": "Fresco",
    "distemper": "Distemper",
    "watercolour": "Watercolour",
    "watercolor": "Watercolour",
    "ink": "Ink",
    "gouache": "Gouache",
    "pastel": "Pastel",
    "charcoal": "Charcoal",
    "graphite": "Graphite",
    "pencil": "Pencil",
    "acrylic": "Acrylic",
    "mixed media": "Mixed media",
}

# Supports: we intentionally collapse "oak/pine/poplar/panel/board" to wood, etc.
SUPPORT_ALIASES = {
    "canvas": "canvas",
    "paper": "paper",
    "cardboard": "cardboard",
    "wood": "wood",
    "panel": "wood",
    "wood panel": "wood",
    "oak": "wood",
    "oak panel": "wood",
    "poplar": "wood",
    "poplar panel": "wood",
    "pine": "wood",
    "pine panel": "wood",
    "board": "wood",
    "stone": "stone",
    "plaster": "plaster",
    "metal": "metal",
}

# phrases to strip (extra detail, mounting, transferring, size-ish noise)
NOISE_PATTERNS = [
    r"\blaid onto\b.*$",                  # "laid onto board"
    r"\bmounted on\b.*$",                 # "mounted on canvas"
    r"\btransferred from\b.*$",           # "transferred from wood"
    r"\bon sheets? of\b.*$",              # "on sheets of paper ..."
    r"\b\d+(\.\d+)?\s*(x|×)\s*\d+(\.\d+)?\b.*$",  # dimensions like "30 x 40 ..."
    r"\bcm\b.*$|\bmm\b.*$|\bin\b.*$",     # units tail
    r"\(.*?\)",                           # parentheticals
]

PREFIX_PATTERNS = [
    r"^\s*artwork\s+made\s+with\s*:\s*",
    r"^\s*artwork\s+made\s+with\s+",
    r"^\s*materials?\s*:\s*",
]

def _cleanup_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s_low = s.lower()

    # remove prefix like "Artwork made with ..."
    for p in PREFIX_PATTERNS:
        s_low = re.sub(p, "", s_low, flags=re.IGNORECASE).strip()

    # remove trailing noise
    for pat in NOISE_PATTERNS:
        s_low = re.sub(pat, "", s_low, flags=re.IGNORECASE).strip()

    # normalize separators
    s_low = s_low.replace(" on ", " on ")
    s_low = s_low.replace("&", " and ")
    s_low = re.sub(r"\s+", " ", s_low).strip(" ,;:-")
    return s_low

def _find_medium(s_low: str) -> str | None:
    # try longest keys first (e.g. "mixed media")
    for k in sorted(MEDIUM_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", s_low):
            return MEDIUM_ALIASES[k]
    return None

def _find_support(s_low: str) -> str | None:
    # prefer explicit "on <support>"
    m = re.search(r"\bon\s+(.+)$", s_low)
    if m:
        tail = m.group(1).strip()
        # keep only first chunk before commas/semicolons
        tail = re.split(r"[;,/]", tail)[0].strip()
        # map to canonical support
        for k in sorted(SUPPORT_ALIASES.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(k)}\b", tail):
                return SUPPORT_ALIASES[k]
        # fallback: if unknown, keep the raw tail (short)
        return tail

    # if no "on", it might be "oak panel", "poplar panel", "paper", etc.
    for k in sorted(SUPPORT_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", s_low):
            return SUPPORT_ALIASES[k]
    return None

def canonical_material(material_text: str) -> str:
    """
    Returns strings like:
      - "Artwork made with Oil on wood"
      - "Artwork made with Fresco"
      - "Artwork made with wood" (if only support is present)
    """
    s_low = _cleanup_text(material_text)

    medium = _find_medium(s_low)
    support = _find_support(s_low)

    # Special-case: fresco is usually not "on <support>" in your data
    if medium == "Fresco":
        return "Artwork made with Fresco"

    if medium and support:
        return f"Artwork made with {medium} on {support}"
    if medium and not support:
        return f"Artwork made with {medium}"
    if support and not medium:
        # keep it simple: "Wood", "canvas", "paper", ...
        # if you prefer Title Case here, change support to support.capitalize()
        return f"Artwork made with {support}"
    return "Artwork made with Unknown"

# ---- Apply to your JSON triplets ----
def clean_materials_in_triplets(
    in_path: str,
    out_path: str | None = None,
    material_link_value: str = "material",
):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        if row.get("link") == material_link_value:
            row["item2_original"] = row["item2"]
            row["item2"] = canonical_material(row["item2"])

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data

# Optional quick sanity check (counts of canonical materials):
def material_counts(triplets):
    c = Counter()
    for r in triplets:
        if r.get("link") == "material":
            c[r["item2"]] += 1
    return c

def analyze_materials(triplets, label):
    materials = [
        r["item2"]
        for r in triplets
        if r.get("link") == "material"
    ]
    unique_materials = sorted(set(materials))
    
    print(f"\n--- {label} ---")
    print(f"Total material entries: {len(materials)}")
    print(f"Number of unique materials: {len(unique_materials)}")
    print("First 20 unique materials:")
    for m in unique_materials[:20]:
        print("  -", m)
    
    return unique_materials


# ---- Run full pipeline ----
in_path = "SemArt/triplets_semart_test.json"
out_path = "SemArt/triplets_semart_test.json"

# Load original
with open(in_path, "r", encoding="utf-8") as f:
    original_data = json.load(f)

# Analyze BEFORE
analyze_materials(original_data, "BEFORE CLEANING")

# Clean
cleaned_data = clean_materials_in_triplets(
    in_path=in_path,
    out_path=out_path
)

# Analyze AFTER
analyze_materials(cleaned_data, "AFTER CLEANING")

