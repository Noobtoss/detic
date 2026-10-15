import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/CenterNet2"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from centernet.config import add_centernet_config
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detic.config import add_detic_config

from load_coco_datasets import load_coco_datasets


def _classifier(classes, output, classifier, resume):
    if resume:
        if not (output / "last_checkpoint").is_file():
            raise ValueError("No checkpoint to resume in {}".format(output))
        if json.loads((output / "classes.json").read_text()) != classes:
            raise ValueError("Resuming requires the same classes in the same order.")
        classifier = output / "classifier.npy"
    if classifier is not None:
        features = np.load(str(classifier))
    else:
        import clip

        encoder, _ = clip.load("ViT-B/32", device="cpu")
        encoder.eval()
        prompts = ["a " + name.replace("_", " ") for name in classes]
        with torch.no_grad():
            features = torch.cat([
                encoder.encode_text(clip.tokenize(prompts[start:start + 256]))
                for start in range(0, len(prompts), 256)
            ]).float().cpu().numpy()
    if features.shape != (len(classes), 512) or not np.isfinite(features).all():
        raise ValueError("classifier must be a finite (number_of_classes, 512) .npy array")
    path = output / "classifier.npy"
    np.save(str(path), features.astype(np.float32))
    (output / "classes.json").write_text(json.dumps(classes, indent=2))
    return str(path)


def _register(data):
    train_names, ann_types = [], []
    for split, group in (("train", "detection"), ("train", "classification"), ("test", "detection")):
        records = data[split][group]
        if not records:
            continue
        name = "custom_{}_{}".format(split, group)
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        DatasetCatalog.register(name, lambda items=records: items)
        MetadataCatalog.get(name).set(thing_classes=data["classes"], evaluator_type="coco")
        if split == "train":
            train_names.append(name)
            ann_types.append("box" if group == "detection" else "image")
    return train_names, ann_types


def build_config(args):
    """Return a frozen Detectron2 config ready for run_train(config)."""
    opts = list(args.opts or [])
    if opts[:1] == ["--"]:
        opts = opts[1:]
    if len(opts) % 2:
        raise ValueError("Config overrides must be KEY VALUE pairs.")
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0:
        raise ValueError("epochs, batch, and imgsz must be positive")

    data = load_coco_datasets(args.train_json, args.test_json, imgs_dir=args.imgs_dir)
    if not any(data["train"].values()) or not data["classes"]:
        raise ValueError("The training input contains no images or categories.")
    train_names, ann_types = _register(data)
    steps_per_epoch = max(1, math.ceil(sum(len(records) for records in data["train"].values()) / args.batch))
    output = Path(args.work_dir) if args.work_dir is not None else ROOT / "runs" / args.run_name

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.RESUME = args.resume
    cfg.merge_from_file(str(args.model_cfg))
    cfg.MODEL.WEIGHTS = str(Path(args.weights).expanduser().resolve())
    cfg.MODEL.DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.TIMM.PRETRAINED = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(data["classes"])
    cfg.MODEL.CENTERNET.NUM_CLASSES = len(data["classes"])
    cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = True
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False
    cfg.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS = "max_size"
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = ""
    cfg.MODEL.DYNAMIC_CLASSIFIER = False
    cfg.MODEL.RESET_CLS_TESTS = False
    cfg.MODEL.WITH_CAPTION = False
    cfg.MODEL.SYNC_CAPTION_BATCH = False
    cfg.MODEL.DATASET_LOSS_WEIGHT = []
    cfg.WITH_IMAGE_LABELS = True
    cfg.FP16 = False
    cfg.DATASETS.TRAIN = tuple(train_names)
    cfg.DATASETS.TEST = ("custom_test_detection",) if data["test"]["detection"] else ()
    cfg.DATALOADER.SAMPLER_TRAIN = "MultiDatasetSampler"
    cfg.DATALOADER.DATASET_ANN = ann_types
    cfg.DATALOADER.DATASET_RATIO = [1] * len(train_names)
    cfg.DATALOADER.MULTI_DATASET_GROUPING = True
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.USE_DIFF_BS_SIZE = False
    cfg.DATALOADER.USE_TAR_DATASET = False
    cfg.DATALOADER.USE_RFS = [False] * len(train_names)
    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.INPUT.CUSTOM_AUG = "ResizeShortestEdge"
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.MIN_SIZE_TRAIN = (args.imgsz,)
    cfg.INPUT.MAX_SIZE_TRAIN = args.imgsz
    cfg.INPUT.MIN_SIZE_TEST = args.imgsz
    cfg.INPUT.MAX_SIZE_TEST = args.imgsz
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.MAX_ITER = math.ceil(args.epochs * steps_per_epoch)
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = min(1000, cfg.SOLVER.MAX_ITER // 10)
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.CHECKPOINT_PERIOD = steps_per_epoch
    cfg.TEST.EVAL_PERIOD = steps_per_epoch if cfg.DATASETS.TEST else 0
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.OUTPUT_DIR = str(output)
    cfg.merge_from_list(opts)
    output = Path(cfg.OUTPUT_DIR).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(output)
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = _classifier(
        data["classes"], output,
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH or args.classifier, cfg.RESUME,
    )
    cfg.freeze()
    (output / "config.yaml").write_text(cfg.dump())
    return cfg
