import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--base_folder", type=str, default="data")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--siglip_model", type=str, default="ViT-SO400M-14-SigLIP")
    p.add_argument("--siglip_pretrained", type=str, default="webli")

    p.add_argument("--finetune_siglip", action="store_true")
    p.add_argument("--siglip_train_triplets", type=str, default=None)
    p.add_argument("--siglip_val_triplets", type=str, default=None)
    p.add_argument("--siglip_out", type=str, default=None)
    p.add_argument("--siglip_run_name", type=str, default=None)
    p.add_argument("--siglip_ckpt", type=str, default=None)

    p.add_argument("--siglip_epochs", type=int, default=1)
    p.add_argument("--siglip_lr", type=float, default=1e-5)
    p.add_argument("--siglip_wd", type=float, default=0.1)
    p.add_argument("--siglip_batch_size", type=int, default=32)
    p.add_argument("--siglip_workers", type=int, default=4)
    p.add_argument("--siglip_log_every", type=int, default=100)
    p.add_argument(
        "--siglip_precision",
        type=str,
        default="fp16",
        choices=["amp", "fp16", "bf16", "fp32"],
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_divisor", type=int, default=4000)
    p.add_argument("--save_embeds", action="store_true")

    p.add_argument(
        "--clip_script",
        type=str,
        default="competitors/clip_ft.py",
        help="Path to your existing CLIP finetune/eval script.",
    )

    return p.parse_args()


def find_project_root() -> Path:
    """
    Assumes this file lives in:
      <project_root>/competitors/siglip_finetune.py
    """
    return Path(__file__).resolve().parents[1]


def main():
    args = parse_args()

    project_root = find_project_root()
    clip_script = project_root / args.clip_script

    if not clip_script.exists():
        raise FileNotFoundError(f"Could not find CLIP script: {clip_script}")

    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(project_root)
        if not existing_pythonpath
        else str(project_root) + os.pathsep + existing_pythonpath
    )

    cmd = [
        sys.executable,
        str(clip_script),
        "--dataset", args.dataset,
        "--base_folder", args.base_folder,
        "--clip_model", args.siglip_model,
        "--clip_pretrained", args.siglip_pretrained,
        "--device_id", str(args.device_id),
        "--clip_epochs", str(args.siglip_epochs),
        "--clip_lr", str(args.siglip_lr),
        "--clip_wd", str(args.siglip_wd),
        "--clip_batch_size", str(args.siglip_batch_size),
        "--clip_workers", str(args.siglip_workers),
        "--clip_log_every", str(args.siglip_log_every),
        "--clip_precision", args.siglip_precision,
        "--seed", str(args.seed),
        "--batch_divisor", str(args.batch_divisor),
    ]

    if args.finetune_siglip:
        cmd.append("--finetune_clip")

    if args.siglip_train_triplets:
        cmd += ["--clip_train_triplets", args.siglip_train_triplets]

    if args.siglip_val_triplets:
        cmd += ["--clip_val_triplets", args.siglip_val_triplets]

    if args.siglip_out:
        cmd += ["--clip_out", args.siglip_out]

    if args.siglip_run_name:
        cmd += ["--clip_run_name", args.siglip_run_name]

    if args.siglip_ckpt:
        cmd += ["--clip_ckpt", args.siglip_ckpt]

    if args.save_embeds:
        cmd.append("--save_embeds")

    subprocess.run(
        cmd,
        check=True,
        cwd=str(project_root),
        env=env,
    )


if __name__ == "__main__":
    main()