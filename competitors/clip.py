import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from src.metrics import *
from src.utils import *
from src.data import *
from src.utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g. Hertziana, SemArt)"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="clip",
        choices=["clip", "siglip", "longclip"],
        help="Vision-language model backend"
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vlm(model_type: str, device: str):
    model_type = model_type.lower()

    if model_type == "clip":
        import open_clip

        model_name = "ViT-B-32"
        pretrained = "laion2b_s34b_b79k"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

        tokenizer = open_clip.get_tokenizer(model_name)

        model = model.to(device).eval()
        return model, preprocess, tokenizer

    if model_type == "siglip":
        import open_clip

        model_name = "ViT-SO400M-14-SigLIP"
        pretrained = "webli"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

        tokenizer = open_clip.get_tokenizer(model_name)

        model = model.to(device).eval()
        return model, preprocess, tokenizer

    if model_type == "longclip":
        from model import longclip  # Long-CLIP repo

        ckpt = os.environ.get(
            "LONGCLIP_CKPT",
            "./checkpoints/longclip-B.pt"
        )

        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"LongCLIP checkpoint missing: {ckpt}"
            )

        model, preprocess = longclip.load(ckpt, device=device)

        tokenizer = lambda texts: longclip.tokenize(texts)

        model = model.to(device).eval()
        return model, preprocess, tokenizer

    raise ValueError(f"Unknown model_type: {model_type}")

def main():

    args = parse_args()

    dataset_name = args.dataset
    model_type = args.model_type
    verbose = args.verbose

    print(f"\nDataset: {dataset_name}")
    print(f"Model:   {model_type}")

    device = get_device()
    print(f"Device:  {device}")

    seed_everything(42)


    triplets = f"data/{dataset_name}/triplets_{dataset_name.lower()}_test.json"
    loaded_data = load_json(triplets)

    print(f"Loaded {len(loaded_data)} triplets")

    model, preprocess, tokenizer = load_vlm(model_type, device)
    

    test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(
        loaded_data,
        preprocess,
        tokenizer,
        base_folder="data",
        split="test",
        dataset_name=dataset_name,
    )

    test_graph_data = test_graph_data.to(device)

    test_dataset = GraphEdgeDataset(test_graph_data, device=device)

    num_batches = max(1, len(test_dataset) // 3000)
    batch_size = max(1, len(test_dataset) // num_batches)

    print(f"Using {num_batches} batches")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    clip_images = []
    clip_texts = []

    for batch in test_loader:

        with torch.no_grad():

            x_img, x_text, edge_index, edge_attr = process_batch(
                batch,
                split="sheaf",
                check_images_=True
            )

            x_img = x_img.to(device)
            x_text = x_text.to(device)

            img_emb = model.encode_image(x_img)
            txt_emb = model.encode_text(x_text)

            img_emb = img_emb[edge_index[0]]
            txt_emb = txt_emb[edge_index[1]]

            clip_images.append(F.normalize(img_emb, dim=1))
            clip_texts.append(F.normalize(txt_emb, dim=1))

    clip_images = torch.cat(clip_images).cpu().numpy()
    clip_texts = torch.cat(clip_texts).cpu().numpy()

    print("Embeddings extracted.")

    results = compute_test_metrics(clip_images, clip_texts, loaded_data, verbose=False, img_path=f'')
    
    recalls = {}
    print('\nEvaluation metrics:')
    for key, value in results.items():
        if 'recall' in key and 'mean' not in key:
            print(f"{key}: {value}")
            recalls[str(key)] = float(value)
    
    save_results(args.model_type, args.dataset, 'retrieval', 42, recalls)

if __name__ == "__main__":
    data_download()
    main()
