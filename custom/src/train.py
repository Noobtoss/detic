import argparse
import sys
import warnings
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

if __package__:
    from .load_coco_datasets import load_coco_datasets
    from .run_training import run_training
else:
    from load_coco_datasets import load_coco_datasets
    from run_training import run_training


DEFAULT_ARGS = {
    "detic-swinb": Namespace(
        run_name="custom_detic",
        work_dir=None,
        model_cfg="/Users/noobtoss/code_nexus/detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        weights="/Users/noobtoss/code_nexus/detic/checkpoints/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        train_json=["/Users/noobtoss/code_nexus/detic/datasets/03Detic/annotations/annotation_train.json"],
        test_json=["/Users/noobtoss/code_nexus/detic/datasets/03Detic/annotations/annotation_test.json"],
        imgs_dir="/Users/noobtoss/code_nexus/detic/datasets/03Detic/Images",
        epochs=4,
        batch=2,
        imgsz=640,
        lr=0.0001,
        workers=2,
        device="auto",
        resume=False,
        classifier=None,
        opts=[],
    ),
}


def train(config):
    return run_training(**config)


def parse_args():
    parser = argparse.ArgumentParser("Detic train parser")
    parser.add_argument("--run_name", type=str, help="Run name; defaults to output/<run_name>")
    parser.add_argument("--work_dir", type=Path, help="Explicit output directory")
    parser.add_argument("--model_cfg", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--train_json", type=Path, nargs="+", help="Training COCO JSON files")
    parser.add_argument("--test_json", type=Path, nargs="*", help="Test COCO JSON files; empty disables evaluation")
    parser.add_argument("--imgs_dir", type=Path)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--classifier", type=Path, help="Precomputed CLIP classifier .npy file")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="Detectron2 config KEY VALUE pairs; use -- to separate them from file lists",
    )
    parser.set_defaults(**deepcopy(vars(DEFAULT_ARGS["detic-swinb"])))
    return parser.parse_args()


def build_config(args):
    return {
        "data": load_coco_datasets(
            train=args.train_json, test=args.test_json, imgs_dir=args.imgs_dir,
        ),
        "config": args.model_cfg,
        "weights": args.weights,
        "output": args.work_dir if args.work_dir is not None else Path(__file__).resolve().parents[2] / "runs" / args.run_name,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr": args.lr,
        "workers": args.workers,
        "device": args.device,
        "resume": args.resume,
        "classifier": args.classifier,
    }


def update_config(config, opts):
    opts = list(opts or [])
    if opts[:1] == ["--"]:
        opts = opts[1:]
    if len(opts) % 2:
        raise ValueError("Config overrides must be KEY VALUE pairs.")
    # run_training.py merges these into Detectron2's config after applying run defaults.
    config["opts"] = opts
    return config


def main():
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        warnings.warn("Running with hardcoded test args")
        args = deepcopy(DEFAULT_ARGS["detic-swinb"])

    config = build_config(args)
    config = update_config(config, args.opts)
    return train(config)


if __name__ == "__main__":
    main()
