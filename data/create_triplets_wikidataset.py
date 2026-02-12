#!/usr/bin/env python3
"""
Build WikiArt-style triplets:
  - item1: image filepath (WIKIART_sample/<filename>)
  - link:  metadata field name OR mapped extracted field label
  - item2: templated text for metadata; raw text for extracted JSON fields

Key fix vs prior version:
- If Qxxxx.json contains {"value": "{...json...}"} (JSON-in-a-string), parse inner JSON
  and emit one triplet per inner key instead of a single "value" blob.
- Apply plausible key-to-label mapping for artist + movement fields.
"""

from __future__ import annotations

import json
import random
import argparse
import os
from collections import defaultdict


import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

QID_RE = re.compile(r"(Q\d+)")


# -------------------------
# Normalization helpers
# -------------------------
def normalize_name(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def norm_key(s: Any) -> str:
    """More aggressive normalization for field matching."""
    if s is None:
        return ""
    s = str(s).casefold().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[_\-/]", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_qid(wikidata_url_or_id: Any) -> Optional[str]:
    if wikidata_url_or_id is None:
        return None
    m = QID_RE.search(str(wikidata_url_or_id))
    return m.group(1) if m else None


def safe_number_to_str(v: Any) -> str:
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v)
    return str(v).strip()


# -------------------------
# Metadata templating
# -------------------------
def build_metadata_text(field: str, value: Any) -> Optional[str]:
    v = safe_number_to_str(value)
    if not v:
        return None

    field_lc = field.strip()

    templates = {
        "artist": lambda x: f"An artwork by {x}.",
        "title": lambda x: f"The artwork is titled “{x}”.",
        "date": lambda x: f"The artwork was created in {x}.",
        "genre": lambda x: f"The genre is {x}.",
        "artwork_style": lambda x: f"The artwork style is {x}.",
        "type": lambda x: f"The artwork type is {x}.",
    }

    if field_lc in templates:
        return templates[field_lc](v)

    pretty_key = field_lc.replace("_", " ").strip().title()
    return f"{pretty_key}: {v}."


# -------------------------
# JSON loading + expansion
# -------------------------
def load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

import json
import re
from typing import Any, Dict, Optional

# matches "Key": "Value" allowing escaped quotes inside value; tolerant-ish
_KV_RE = re.compile(
    r'"(?P<key>(?:\\.|[^"\\])*)"\s*:\s*"(?P<val>(?:\\.|[^"\\])*)"',
    re.DOTALL
)

def _unescape_json_string(s: str) -> str:
    """
    Best-effort unescape for JSON-style escaped sequences.
    If it fails, return the original.
    """
    try:
        # Wrap as a JSON string so json.loads can unescape sequences safely
        return json.loads(f'"{s}"')
    except Exception:
        return s

def parse_kv_pairs_loose(s: str) -> Optional[Dict[str, str]]:
    """
    Extract key/value pairs from a pseudo-JSON string even if the object is truncated.
    Returns dict of extracted pairs, or None if nothing found.
    """
    pairs = {}
    for m in _KV_RE.finditer(s):
        k_raw = m.group("key")
        v_raw = m.group("val")
        k = _unescape_json_string(k_raw).strip()
        v = _unescape_json_string(v_raw).strip()
        if k:
            pairs[k] = v
    return pairs or None

def maybe_parse_json_string(s: str) -> Optional[Any]:
    """
    Robust parser for JSON stored inside a string.
    Handles:
      - valid JSON object/list strings
      - wrapper junk before/after the JSON
      - truncated JSON by falling back to regex key/value extraction
    """
    if not isinstance(s, str):
        return None

    t = s.strip()
    if not t:
        return None

    # 1) strict parse (fast path)
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            return json.loads(t)
        except Exception:
            pass

    # 2) salvage: take substring between first '{' and last '}' (if present)
    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        sub = t[i:j+1]
        try:
            return json.loads(sub)
        except Exception:
            # continue to loose extraction
            loose = parse_kv_pairs_loose(sub)
            if loose is not None:
                return loose

    # 3) if it looks like it *contains* an object but is truncated, attempt loose extraction on whole string
    if "{" in t and '"' in t and ":" in t:
        loose = parse_kv_pairs_loose(t)
        if loose is not None:
            return loose

    return None


def unwrap_extracted_payload(obj: Any) -> Any:
    """
    Many of your Qxxxx.json appear to be like:
      {"value": "{ ... inner json ... }"}
    or sometimes {"value": {...}} or directly {...}.
    This function returns the inner dict if possible.
    """
    if obj is None:
        return None

    # If it's already a dict with multiple meaningful keys, keep as-is
    if isinstance(obj, dict):
        # common wrapper case: single key 'value'
        if set(obj.keys()) == {"value"}:
            v = obj["value"]
            if isinstance(v, dict) or isinstance(v, list):
                return v
            if isinstance(v, str):
                parsed = maybe_parse_json_string(v)
                return parsed if parsed is not None else v
            return v

        # Another wrapper possibility: {"data": "..."} etc. (keep generic)
        # But if any value is a JSON string dict, prefer that dict.
        # Only do this if it seems like the dict is mostly wrapper-y.
        for k in ("value", "data", "result", "output"):
            if k in obj and isinstance(obj[k], str):
                parsed = maybe_parse_json_string(obj[k])
                if isinstance(parsed, dict):
                    return parsed

        return obj

    # If it's a string containing JSON, parse it
    if isinstance(obj, str):
        parsed = maybe_parse_json_string(obj)
        return parsed if parsed is not None else obj

    return obj


def iter_text_fields(obj: Any) -> Iterable[Tuple[str, str]]:
    """
    Yield (key, text_value) from an extracted object.
    - If dict: iterate items
    - If list: yield indexed items
    - If scalar: yield ("value", str(obj))
    """
    if obj is None:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, str):
                txt = v.strip()
            elif isinstance(v, list):
                parts = [str(it).strip() for it in v if it is not None and str(it).strip()]
                txt = "; ".join(parts)
            else:
                txt = str(v).strip()
            if txt:
                yield str(k), txt
        return

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            if v is None:
                continue
            txt = str(v).strip()
            if txt:
                yield f"item_{i}", txt
        return

    txt = str(obj).strip()
    if txt:
        yield "value", txt


# -------------------------
# Plausible key mapping
# -------------------------
ARTIST_LABELS = {
    "life": "artist life",
    "influences": "artist influences",
    "education": "artist education",
    "style and art movement": "artist movement",
    "style and movement": "artist movement",
    "art movement": "artist movement",
    "movement": "artist movement",
    "time period and place of activity": "artist time and place of activity",
    "time and place of activity": "artist time and place of activity",
    "content and subjects of their artworks": "artist content and subjects",
    "content and subjects": "artist content and subjects",
    "materials and techniques used for the artworks": "artist materials and techniques",
    "materials and techniques": "artist materials and techniques",
}

MOVEMENT_LABELS = {
    "definition overview": "style overview",
    "definition / overview": "style overview",
    "overview": "style overview",
    "historical context": "style historical context",
    "key characteristics": "style key characteristics",
    "influences and precursors": "style influences and precursors",
    "influences precursors": "style influences and precursors",
    "major figures or artists": "style major figures",
    "major figures": "style major figures",
    "notable works": "style notable works",
    "subtypes or related movements": "style related movements",
    "related movements": "style related movements",
    "reception and legacy": "style reception as legacy",
    "reception legacy": "style reception as legacy",
    "geographical and temporal spread": "style geographical and temporal evolution",
    "geographical temporal spread": "style geographical and temporal evolution",
    "geographical and temporal evolution": "style geographical and temporal evolution",
    "manifestos or theoretical writings": "style manifestos",
    "manifestos": "style manifestos",
    "crossdisciplinary influence": "style cross-disciplinary influence",
    "cross disciplinary influence": "style cross-disciplinary influence",
    "criticism or controversy": "style criticisms and controversy",
    "criticisms and controversy": "style criticisms and controversy",
}

def plausible_map_key(domain: str, raw_key: str) -> Optional[str]:
    """
    Map raw extracted keys to your requested label set, but only if plausible.
    Strategy:
      - normalize raw_key
      - try exact match against normalized variants in the mapping tables
      - try containment-based match (e.g., raw contains 'historical context')
    """
    nk = norm_key(raw_key)

    if domain == "artist":
        mapping = ARTIST_LABELS
    elif domain == "movement":
        mapping = MOVEMENT_LABELS
    else:
        return None

    # Build normalized lookup once
    norm_map = {norm_key(k): v for k, v in mapping.items()}

    # Exact normalized match
    if nk in norm_map:
        return norm_map[nk]

    # Containment-based match (plausible)
    # Only accept if the raw key contains one of the canonical phrases
    # and the phrase is reasonably distinctive (>= 4 chars).
    best = None
    best_len = 0
    for canon_k, label in norm_map.items():
        if len(canon_k) < 4:
            continue
        if canon_k in nk:
            if len(canon_k) > best_len:
                best = label
                best_len = len(canon_k)

    return best


# -------------------------
# CSV joins
# -------------------------
def build_lookup_map_artists(artists_df: pd.DataFrame) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for _, row in artists_df.iterrows():
        name_norm = normalize_name(row.get("input_name"))
        qid = extract_qid(row.get("wikidata"))
        if name_norm and qid:
            m[name_norm] = qid
    return m


def build_lookup_map_movements(mov_df: pd.DataFrame) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for _, row in mov_df.iterrows():
        name_norm = normalize_name(row.get("movement_name"))
        qid = extract_qid(row.get("wikidata"))
        if name_norm and qid:
            m[name_norm] = qid
    return m


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wikiart_csv", default="WIKIART_sample.csv")
    parser.add_argument("--artists_csv", default="list_of_artists.csv")
    parser.add_argument("--movements_csv",  default="list_of_art_movements.csv")
    parser.add_argument("--image_dir", help="Folder containing images named like the CSV filename column.", default="WIKIART_sample")
    parser.add_argument("--json_dir", help="Folder containing Qxxxx.json files.", default=".")
    parser.add_argument("--out_json", default="triplets_wikiart_sample.json")
    parser.add_argument("--relative_paths", action="store_true",
                    help="If set, item1 paths are relative: WIKIART_sample/<filename> instead of absolute.", default=True)
    parser.add_argument("--skip_fields", nargs="*", default=["filename", 'pixelsx', 'pixelsy', 'size_bytes', 'source'],
                    help="Metadata fields to skip (default: filename).")
    args = parser.parse_args()

    wiki_df = pd.read_csv(args.wikiart_csv)
    artists_df = pd.read_csv(args.artists_csv)
    mov_df = pd.read_csv(args.movements_csv)

    artist_map = build_lookup_map_artists(artists_df)
    movement_map = build_lookup_map_movements(mov_df)

    triplets: List[Dict[str, str]] = []
    missing_artist = 0
    missing_movement = 0
    missing_json = 0

    skip = set(args.skip_fields or [])

    for _, row in wiki_df.iterrows():
        filename = str(row.get("filename", "")).strip()
        if not filename or filename.lower() == "nan":
            continue

        if args.relative_paths:
            rel_folder = os.path.basename(os.path.normpath(args.image_dir))
            item1 = os.path.join(rel_folder, filename)
        else:
            item1 = os.path.join(args.image_dir, filename)

        # --- metadata -> templated text ---
        for col in wiki_df.columns:
            if col in skip or col == "filename":
                continue
            val = row.get(col)
            if isinstance(val, float) and pd.isna(val):
                continue
            text = build_metadata_text(col, val)
            if text:
                triplets.append({"item1": item1, "link": col, "item2": text})

        # --- artist extracted fields ---
        artist_name_norm = normalize_name(row.get("artist"))
        qid_artist = artist_map.get(artist_name_norm)
        if qid_artist:
            p = os.path.join(args.json_dir, 'jsons_artist_summaries', f"{qid_artist}.json")
            raw = load_json(p)
            if raw is None:
                missing_json += 1
            else:
                payload = unwrap_extracted_payload(raw)
                # If payload is still a JSON string dict, parse once more
                if isinstance(payload, str):
                    parsed = maybe_parse_json_string(payload)
                    if parsed is not None:
                        payload = parsed

                for k, txt in iter_text_fields(payload):
                    mapped = plausible_map_key("artist", k)
                    link = mapped if mapped is not None else f"artist.{k}"
                    triplets.append({"item1": item1, "link": link, "item2": txt})
        else:
            if artist_name_norm:
                missing_artist += 1

        # --- movement extracted fields (from artwork_style) ---
        movement_norm = normalize_name(row.get("artwork_style"))
        qid_mov = movement_map.get(movement_norm)
        if qid_mov:
            p = os.path.join(args.json_dir, 'jsons_movement_summaries', f"{qid_mov}.json")
            raw = load_json(p)
            if raw is None:
                missing_json += 1
            else:
                payload = unwrap_extracted_payload(raw)
                if isinstance(payload, str):
                    parsed = maybe_parse_json_string(payload)
                    if parsed is not None:
                        payload = parsed

                for k, txt in iter_text_fields(payload):
                    mapped = plausible_map_key("movement", k)
                    link = mapped if mapped is not None else f"movement.{k}"
                    triplets.append({"item1": item1, "link": link, "item2": txt})
        else:
            if movement_norm:
                missing_movement += 1

    # De-duplicate exact triplets
    seen = set()
    deduped: List[Dict[str, str]] = []
    for t in triplets:
        key = (t["item1"], t["link"], t["item2"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(deduped)} triplets to: {args.out_json}")
    print(f"Missing artist join: {missing_artist}")
    print(f"Missing movement join: {missing_movement}")
    print(f"Missing JSON files: {missing_json}")


    random.seed(args.seed)

    # -------------------------
    # Load triplets
    # -------------------------
    with open(args.input_json, "r", encoding="utf-8") as f:
        triplets = json.load(f)

    # -------------------------
    # Group by image (item1)
    # -------------------------
    grouped = defaultdict(list)
    for t in triplets:
        grouped[t["item1"]].append(t)

    images = list(grouped.keys())
    random.shuffle(images)

    # -------------------------
    # Compute split indices
    # -------------------------
    n_total = len(images)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # -------------------------
    # Collect triplets
    # -------------------------
    train = []
    val = []
    test = []

    for img in train_imgs:
        train.extend(grouped[img])

    for img in val_imgs:
        val.extend(grouped[img])

    for img in test_imgs:
        test.extend(grouped[img])

    # -------------------------
    # Save
    # -------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "triplets_wikidataset_train.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "triplets_wikidataset_val.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "triplets_wikidataset_test.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Images total: {n_total}")
    print(f"Train images: {len(train_imgs)}")
    print(f"Val images: {len(val_imgs)}")
    print(f"Test images: {len(test_imgs)}")
    print()
    print(f"Train triplets: {len(train)}")
    print(f"Val triplets: {len(val)}")
    print(f"Test triplets: {len(test)}")


if __name__ == "__main__":
    main()
    


