import math
import os
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import yaml

try:
    import wandb as wb
    if not hasattr(wb, "__version__"):
        wb = None
except ImportError:
    wb = None

LOG_EVERY_N_STEPS = 40
_step = 0


def run_callbacks(callbacks, event, **kwargs):
    handlers = (callbacks or {}).get(event, [])
    if callable(handlers):
        handlers = [handlers]
    for callback in handlers:
        callback(**kwargs)


def _flatten(metrics, prefix, keep_path=True):
    values = {}
    for name, value in metrics.items():
        key = "{}/{}".format(prefix, name).lower()
        if isinstance(value, Mapping):
            values.update(_flatten(value, key if keep_path else prefix, keep_path))
        elif isinstance(value, Real) and math.isfinite(value):
            values[key] = float(value)
    return values


def on_pretrain_routine_start(cfg, model, resume=False):
    """Start W&B, reusing the saved run ID when resuming this output directory."""
    global _step
    if wb is None:
        return

    _step = 0
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    id_file = output / "wandb_run_id.txt"
    if not wb.run:
        run_id = id_file.read_text().strip() if resume and id_file.is_file() else None
        wb.init(
            project=os.environ.get("WANDB_PROJECT", "runs-detic"),
            name=os.environ.get("WANDB_NAME", output.name),
            dir=str(output),
            config=yaml.safe_load(cfg.dump()),
            id=run_id,
            resume="allow" if run_id else None,
        )
    wb.run.define_metric("trainer/step")
    wb.run.define_metric("*", step_metric="trainer/step")
    id_file.write_text(wb.run.id + "\n")


def on_train_step_end(cfg, model, step, metrics):
    """Log every N steps, plus the first and last step (1-based)."""
    global _step
    if wb and wb.run:
        _step = step
        max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER
        if step == 1 or step % LOG_EVERY_N_STEPS == 0 or step == max_iter:
            wb.run.log({**_flatten(metrics, "train"), "trainer/step": step})


def on_test_end(cfg, model, results):
    """Log filtered evaluation metrics as lowercase names under eval/."""
    if wb and wb.run:
        values = _flatten(results, "eval", keep_path=False)
        if values:
            wb.run.log({**values, "trainer/step": _step})


def on_train_end(cfg, model, exit_code=0):
    """Finish when do_train completes, before the separate final evaluation."""
    if wb and wb.run:
        wb.run.finish(exit_code=exit_code)


wb_callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_step_end": on_train_step_end,
        "on_test_end": on_test_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
