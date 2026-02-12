import pandas as pd
import csv
import os
import json
import random
from collections import defaultdict
import csv

def load_tsv_safe(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows

def assign(file_name: str, prefix: str):
    counter = 0
    output = []
    
    # Mappatura vecchi nomi -> nuovi nomi
    field_mapping = {
        "paragraph foto EN": "photography",
        "paragraph obj EN": "object",
        "paragraph verwalter EN": "context",
        "paragraph iconclass EN": "iconography",
        "artist": "artist",
        "medium short en": "medium",
        "period_50yr": "acquisition period",
    }
    
    useful_fields = list(field_mapping.keys())
    
    # Lunghezza massima per le descrizioni
    MAX_LENGTH = 1e9

    if not file_name.endswith("sv"):
        print(f"[SKIP] Non TSV/CSV: {file_name}")
        return output

    sep = "\t" if file_name.endswith(".tsv") else ","
    id_field = "BILDDATEI-NR. (POSITIV)" if file_name.endswith('.tsv') else "filename"
    
    df = pd.read_csv(
        file_name,
        sep=sep,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="warn"   
    )

    if id_field not in df.columns:
        counter +=1
        print(f"[ERRORE] Colonna '{id_field}' ASSENTE in {file_name}.")
        print("Colonne trovate:", df.columns.tolist())
        return output
    
    images_missing = 0
    skipped_long = 0
    
    for idx, row in df.iterrows():
        image_id = row[id_field]

        if pd.isna(image_id):
            print(f"[WARNING] ID immagine mancante alla riga {idx} nel file {file_name}")
            images_missing +=1
            continue

        item1_value = f"{prefix}/{image_id}.jpg"

        for old_field in useful_fields:
            if old_field in df.columns:
                item2_value = row[old_field]
                if pd.notna(item2_value) and item2_value.strip():
                    # Verifica lunghezza e tronca se necessario
                    if len(item2_value) > MAX_LENGTH:
                        # Opzione 1: Scartare completamente
                        skipped_long += 1
                        continue
                        
                        # Opzione 2: Troncare (attiva per default)
                        # item2_value = item2_value[:MAX_LENGTH].strip()
                        # skipped_long += 1
                    
                    # Usa il nuovo nome del campo
                    new_field_name = field_mapping[old_field]
                    
                    output.append({
                        "item1": item1_value,
                        "item2": item2_value,
                        "link": new_field_name
                    })
    
    if skipped_long > 0:
        print(f"[INFO] {skipped_long} descrizioni troncate a {MAX_LENGTH} caratteri in {file_name}")
    
    return output


def generate_complete_folder(base_folder: str):
    prefix = "zeichnungen" if "zeichnungen" in base_folder else "gemalde"
    complete_results = []

    for item in os.listdir(base_folder):
        file_path = os.path.join(base_folder, item)
        result_partial = assign(file_path, prefix)
        complete_results.extend(result_partial)

    return complete_results


def split_dataset(input_file: str, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset ensuring each image (item1) appears in only one set.
    
    Args:
        input_file: path to the complete JSON file
        train_ratio: percentage for training set (default 70%)
        val_ratio: percentage for validation set (default 15%)
        test_ratio: percentage for test set (default 15%)
        seed: random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data: three lists of records
    """
    
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total records loaded: {len(data)}")
    
    # Group all records by image (item1)
    images_dict = defaultdict(list)
    for record in data:
        images_dict[record['item1']].append(record)
    
    # Get list of all unique image IDs
    all_image_ids = list(images_dict.keys())
    print(f"Number of unique images: {len(all_image_ids)}")
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(all_image_ids)
    
    # Calculate split indices
    n_images = len(all_image_ids)
    train_end = int(n_images * train_ratio)
    val_end = train_end + int(n_images * val_ratio)
    
    # Split image IDs
    train_ids = set(all_image_ids[:train_end])
    val_ids = set(all_image_ids[train_end:val_end])
    test_ids = set(all_image_ids[val_end:])
    
    # Create datasets by assigning all records of each image to the appropriate set
    train_data = []
    val_data = []
    test_data = []
    
    for image_id, records in images_dict.items():
        if image_id in train_ids:
            train_data.extend(records)
        elif image_id in val_ids:
            val_data.extend(records)
        elif image_id in test_ids:
            test_data.extend(records)
    
    # Print statistics
    print("\n=== SPLIT STATISTICS ===")
    print(f"Train: {len(train_ids)} images, {len(train_data)} records ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Val:   {len(val_ids)} images, {len(val_data)} records ({len(val_data)/len(data)*100:.1f}%)")
    print(f"Test:  {len(test_ids)} images, {len(test_data)} records ({len(test_data)/len(data)*100:.1f}%)")
    
    # Verify no overlaps between sets
    assert len(train_ids & val_ids) == 0, "Overlap between train and val!"
    assert len(train_ids & test_ids) == 0, "Overlap between train and test!"
    assert len(val_ids & test_ids) == 0, "Overlap between val and test!"
    print("\n✓ Verified: no overlap between sets")
    
    # Save datasets
    with open('triplets_hertziana_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open('triplets_hertziana_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open('triplets_hertziana_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    
    # Save split information for reference
    split_info = {
        'train_ids': list(train_ids),
        'val_ids': list(val_ids),
        'test_ids': list(test_ids),
        'stats': {
            'train': {'n_images': len(train_ids), 'n_records': len(train_data)},
            'val': {'n_images': len(val_ids), 'n_records': len(val_data)},
            'test': {'n_images': len(test_ids), 'n_records': len(test_data)}
        }
    }
    
    with open('split_info.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print("✓ Split info saved in: split_info.json")
    
    return train_data, val_data, test_data


def split_dataset_stratified(input_file: str, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Stratified split maintaining proportions between zeichnungen and gemalde.
    
    Args:
        input_file: path to the complete JSON file
        train_ratio: percentage for training set (default 70%)
        val_ratio: percentage for validation set (default 15%)
        test_ratio: percentage for test set (default 15%)
        seed: random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data: three lists of records
    """
    
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total records loaded: {len(data)}")
    
    # Separate by type (zeichnungen vs gemalde)
    zeichnungen_dict = defaultdict(list)
    gemalde_dict = defaultdict(list)
    
    for record in data:
        if record['item1'].startswith('zeichnungen/'):
            zeichnungen_dict[record['item1']].append(record)
        else:
            gemalde_dict[record['item1']].append(record)
    
    print(f"Zeichnungen images: {len(zeichnungen_dict)}")
    print(f"Gemalde images: {len(gemalde_dict)}")
    
    # Set random seed
    random.seed(seed)
    
    def split_group(images_dict):
        """Helper function to split a group of images"""
        ids = list(images_dict.keys())
        random.shuffle(ids)
        n = len(ids)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return ids[:train_end], ids[train_end:val_end], ids[val_end:]
    
    # Perform separate splits for each type
    z_train, z_val, z_test = split_group(zeichnungen_dict)
    g_train, g_val, g_test = split_group(gemalde_dict)
    
    # Combine the splits
    train_ids = set(z_train + g_train)
    val_ids = set(z_val + g_val)
    test_ids = set(z_test + g_test)
    
    # Merge dictionaries
    all_images = {**zeichnungen_dict, **gemalde_dict}
    
    # Create final datasets
    train_data = [record for img_id in train_ids for record in all_images[img_id]]
    val_data = [record for img_id in val_ids for record in all_images[img_id]]
    test_data = [record for img_id in test_ids for record in all_images[img_id]]
    
    # Print statistics
    print("\n=== STRATIFIED SPLIT STATISTICS ===")
    print(f"Train: {len(train_ids)} images ({len([i for i in train_ids if i.startswith('zeichnungen')])} Z, "
          f"{len([i for i in train_ids if i.startswith('gemalde')])} G), {len(train_data)} records")
    print(f"Val:   {len(val_ids)} images ({len([i for i in val_ids if i.startswith('zeichnungen')])} Z, "
          f"{len([i for i in val_ids if i.startswith('gemalde')])} G), {len(val_data)} records")
    print(f"Test:  {len(test_ids)} images ({len([i for i in test_ids if i.startswith('zeichnungen')])} Z, "
          f"{len([i for i in test_ids if i.startswith('gemalde')])} G), {len(test_data)} records")
    
    # Verify no overlaps
    assert len(train_ids & val_ids) == 0, "Overlap between train and val!"
    assert len(train_ids & test_ids) == 0, "Overlap between train and test!"
    assert len(val_ids & test_ids) == 0, "Overlap between val and test!"
    print("\n✓ Verified: no overlap between sets")
    
    # Save datasets
    with open('train_set.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open('val_set.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open('test_set.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("\n✓ Files saved: train_set.json, val_set.json, test_set.json")
    
    # Save split information
    split_info = {
        'train_ids': list(train_ids),
        'val_ids': list(val_ids),
        'test_ids': list(test_ids),
        'stats': {
            'train': {'n_images': len(train_ids), 'n_records': len(train_data)},
            'val': {'n_images': len(val_ids), 'n_records': len(val_data)},
            'test': {'n_images': len(test_ids), 'n_records': len(test_data)}
        }
    }
    
    with open('split_info.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print("✓ Split info saved in: split_info.json")
    
    return train_data, val_data, test_data


def analyze_split():
    """
    Analyze the distribution of fields across different splits.
    Useful to verify the quality of the split.
    """
    from collections import Counter
    
    for split_name in ['train_set.json', 'val_set.json', 'test_set.json']:
        with open(split_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n=== {split_name.upper()} ===")
        print(f"Total records: {len(data)}")
        print(f"Unique images: {len(set(r['item1'] for r in data))}")
        
        # Distribution by link type
        link_counts = Counter(r['link'] for r in data)
        print("\nDistribution by link type:")
        for link, count in link_counts.most_common():
            print(f"  {link}: {count}")
        
        # Distribution by image type (zeichnungen vs gemalde)
        type_counts = Counter('zeichnungen' if r['item1'].startswith('zeichnungen') 
                             else 'gemalde' for r in data)
        print("\nDistribution by image type:")
        for img_type, count in type_counts.items():
            print(f"  {img_type}: {count}")


def verify_no_leakage():
    """
    Verify that no image appears in multiple splits.
    This is a safety check to ensure data integrity.
    """
    
    # Load all splits
    with open('train_set.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('val_set.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open('test_set.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Extract unique image IDs from each split
    train_images = set(r['item1'] for r in train_data)
    val_images = set(r['item1'] for r in val_data)
    test_images = set(r['item1'] for r in test_data)
    
    print("=== LEAKAGE VERIFICATION ===")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")
    
    # Check for overlaps
    train_val_overlap = train_images & val_images
    train_test_overlap = train_images & test_images
    val_test_overlap = val_images & test_images
    
    if train_val_overlap:
        print(f"❌ ERROR: {len(train_val_overlap)} images in both train and val!")
        print(f"   Examples: {list(train_val_overlap)[:5]}")
    else:
        print("✓ No overlap between train and val")
    
    if train_test_overlap:
        print(f"❌ ERROR: {len(train_test_overlap)} images in both train and test!")
        print(f"   Examples: {list(train_test_overlap)[:5]}")
    else:
        print("✓ No overlap between train and test")
    
    if val_test_overlap:
        print(f"❌ ERROR: {len(val_test_overlap)} images in both val and test!")
        print(f"   Examples: {list(val_test_overlap)[:5]}")
    else:
        print("✓ No overlap between val and test")
    
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("\n✅ ALL CHECKS PASSED: No data leakage detected!")


output = generate_complete_folder('zeichnungen/enriched_data/')
output.extend(generate_complete_folder('gemalde/enriched_data/'))
with open("complete_set.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
    
train, val, test = split_dataset(
        'complete_set.json',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )


