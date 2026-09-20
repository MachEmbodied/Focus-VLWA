"""Unified Focus-VLWA command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path


def _post_training_config(args: argparse.Namespace):
    from focus_vlwa.configs.training import PostTrainingConfig

    return PostTrainingConfig(
        dataset=args.dataset,
        init_checkpoint=args.init_checkpoint,
        output_dir=args.output_dir,
        norm_stats_dir=args.norm_stats_dir,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_steps=args.num_steps,
        save_interval=args.save_interval,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="focus-vlwa")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-checkpoint", help="validate checkpoint names and shapes")
    check.add_argument("checkpoint", type=Path)
    commands.add_parser("install-transformers-patch", help="install the pinned transformers compatibility patch")
    train = commands.add_parser("post-train", help="run Focus-VLWA post-training")
    train.add_argument("--dataset", required=True)
    train.add_argument("--init-checkpoint", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--norm-stats-dir", required=True, type=Path)
    train.add_argument("--tokenizer-path", type=Path)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--num-steps", type=int, default=20_000)
    train.add_argument("--save-interval", type=int, default=2_000)
    args = parser.parse_args()

    if args.command == "check-checkpoint":
        from focus_vlwa.scripts.check_checkpoint import validate_checkpoint

        missing, unexpected, mismatches = validate_checkpoint(args.checkpoint)
        if missing or unexpected or mismatches:
            raise SystemExit(
                f"Checkpoint validation failed: missing={missing[:20]}, unexpected={unexpected[:20]}, "
                f"shape_mismatches={mismatches[:20]}"
            )
        print(f"Checkpoint is compatible: {args.checkpoint}")
    elif args.command == "install-transformers-patch":
        from focus_vlwa.scripts.install_transformers_patch import install_transformers_patch

        install_transformers_patch()
    elif args.command == "post-train":
        from focus_vlwa.post_training.trainer import run_post_training

        run_post_training(_post_training_config(args))


if __name__ == "__main__":
    main()
