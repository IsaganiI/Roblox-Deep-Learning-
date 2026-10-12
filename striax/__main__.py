"""Command-line entry point.

    python -m striax fetch      download the latest prices
    python -m striax train      train the LSTM and save artifacts
    python -m striax predict    predict the next day with the saved model
    python -m striax daily      fetch + predict (trains first only if no model exists)
    python -m striax all        fetch + train + predict
"""

import argparse

from . import config, storage


def _model_exists() -> bool:
    return storage.exists(storage.join(config.MODEL_DIR, config.METADATA_FILE))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="striax", description="STRIAX RBLX prediction pipeline")
    parser.add_argument("command", choices=["fetch", "train", "predict", "daily", "all"])
    cmd = parser.parse_args(argv).command

    # Imported lazily so `fetch` doesn't need PyTorch loaded.
    from .data import run_fetch

    if cmd in ("fetch", "daily", "all"):
        run_fetch()
    if cmd in ("train", "all") or (cmd == "daily" and not _model_exists()):
        from .train import run_train
        run_train()
    if cmd in ("predict", "daily", "all"):
        from .predict import run_predict
        run_predict()


if __name__ == "__main__":
    main()
