import argparse
from argparse import Namespace
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
import train_net

from build_config import build_config
from callbacks import wb_callbacks as callbacks


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

class CustomCheckpointer(DetectionCheckpointer):
    def _load_model(self, checkpoint):
        # Keep this dataset's fixed CLIP vocabulary when loading old weights.
        checkpoint["model"] = {
            key: value for key, value in checkpoint["model"].items()
            if not key.endswith("cls_score.zs_weight")
        }
        incompatible = super()._load_model(checkpoint)
        incompatible.missing_keys[:] = [
            key for key in incompatible.missing_keys
            if not key.endswith("cls_score.zs_weight")
        ]
        return incompatible

def train(config):
    config = config
    output = Path(config.OUTPUT_DIR)
    setup_logger(output=config.OUTPUT_DIR, name="detectron2")
    logger = setup_logger(output=config.OUTPUT_DIR, name="detic.custom")
    logger.info("Training groups: %s; classes: %s", config.DATALOADER.DATASET_ANN, config.MODEL.ROI_HEADS.NUM_CLASSES)

    model = train_net.build_model(config)
    if not config.RESUME:
        CustomCheckpointer(model).load(config.MODEL.WEIGHTS)

    # Initial weights are already loaded with the custom vocabulary preserved.
    # On resume, train_net restores model and optimizer from last_checkpoint.
    train_cfg = config.clone()
    train_cfg.defrost()
    train_cfg.MODEL.WEIGHTS = ""
    train_cfg.freeze()
    train_net.do_train(train_cfg, model, resume=config.RESUME, callbacks=callbacks)
    train_net.do_test(train_cfg, model, callbacks=callbacks)


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


def main():
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        warnings.warn("Running with hardcoded test args")
        args = deepcopy(DEFAULT_ARGS["detic-swinb"])

    config = build_config(args)
    train(config)


if __name__ == "__main__":
    main()
