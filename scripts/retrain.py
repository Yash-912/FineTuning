from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.review_queue import ReviewQueue
from src.data.deduplicator import deduplicate
from src.data.balancer import check_balance, stratified_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("retrain")

CONFIG_DIR = Path("configs")
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/qwen-injection-detector")


def merge_reviewed_labels(review_queue: ReviewQueue, existing_train: pd.DataFrame, existing_val: pd.DataFrame) -> pd.DataFrame:
    labeled_path = DATA_DIR / "reviewed_labels.jsonl"
    n = review_queue.export_labeled(str(labeled_path))
    if n == 0:
        logger.info("No new labeled examples — retraining on existing data only")
        return pd.concat([existing_train, existing_val], ignore_index=True)

    labeled = pd.read_json(str(labeled_path), lines=True)
    labeled["source"] = "human_review"
    logger.info("Loaded %d human-labeled examples", len(labeled))

    combined = pd.concat([existing_train, existing_val, labeled], ignore_index=True)
    logger.info("Combined dataset: %d rows (train=%d, val=%d, labeled=%d)",
                len(combined), len(existing_train), len(existing_val), len(labeled))
    return combined


def run_train(config_path: str = "configs/training_config.yaml") -> bool:
    logger.info("Starting QLoRA training...")
    result = subprocess.run(
        [sys.executable, "scripts/train_qlora.py", "--config", config_path],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("Training failed with exit code %d", result.returncode)
        return False
    logger.info("Training completed successfully")
    return True


def run_calibrate() -> bool:
    logger.info("Running temperature calibration...")
    result = subprocess.run(
        [sys.executable, "scripts/calibrate.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("Calibration failed with exit code %d", result.returncode)
        return False
    logger.info("Calibration completed successfully")
    return True


def copy_to_shadow():
    shadow_path = MODEL_DIR / "shadow"
    best_path = MODEL_DIR / "best"

    if not best_path.exists():
        logger.warning("Best model not found at %s — skipping shadow copy", best_path)
        return

    if shadow_path.exists():
        shutil.rmtree(str(shadow_path))

    shutil.copytree(str(best_path), str(shadow_path))
    logger.info("Shadow model updated: %s -> %s", best_path, shadow_path)


def main():
    import argparse
    parser = argparse.ArgumentParser("retrain")
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--train-config", default="configs/training_config.yaml")
    parser.add_argument("--db", default="data/review_queue.db")
    parser.add_argument("--shadow", action="store_true", help="Copy model to shadow path after training")
    parser.add_argument("--skip-train", action="store_true", help="Only prepare data, skip training")
    args = parser.parse_args()

    start_time = datetime.now(timezone.utc)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("RETRAIN PIPELINE")
    logger.info("=" * 60)

    logger.info("Loading existing training data...")
    train_path = cfg["output"]["dir"] + "/train.parquet"
    val_path = cfg["output"]["dir"] + "/val.parquet"
    test_path = cfg["output"]["dir"] + "/test.parquet"

    existing_train = pd.read_parquet(train_path)
    existing_val = pd.read_parquet(val_path)
    existing_test = pd.read_parquet(test_path)
    logger.info("Existing: train=%d, val=%d, test=%d", len(existing_train), len(existing_val), len(existing_test))

    review_queue = ReviewQueue(args.db)
    combined = merge_reviewed_labels(review_queue, existing_train, existing_val)

    if any(combined.duplicated(subset="text").sum() > 0 for _ in [1]):
        before = len(combined)
        combined = deduplicate(combined, cfg["processing"])
        logger.info("Dedup removed %d rows (%d -> %d)", before - len(combined), before, len(combined))

    logger.info("Checking balance...")
    combined = check_balance(combined, cfg["processing"].get("target_benign_to_injection_ratio", 2.0))

    logger.info("Stratified splitting...")
    new_train, new_val, new_test = stratified_split(combined, cfg["split"])

    backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    for split_name, split_df in [("train", new_train), ("val", new_val), ("test", new_test)]:
        orig_path = Path(cfg["output"]["dir"]) / f"{split_name}.parquet"
        backup_path = Path(cfg["output"]["dir"]) / f"{split_name}_{backup_suffix}.parquet"
        shutil.copy2(str(orig_path), str(backup_path))
        split_df.to_parquet(str(orig_path), index=False)
        logger.info("Updated %s: %d rows (backup: %s)", split_name, len(split_df), backup_path)

    elapsed = datetime.now(timezone.utc) - start_time
    logger.info("Data preparation complete in %s", elapsed)

    if args.skip_train:
        logger.info("Skipping training (--skip-train)")
        return

    if not run_train(args.train_config):
        logger.error("Training failed — restoring backups")
        for split_name in ["train", "val", "test"]:
            orig_path = Path(cfg["output"]["dir"]) / f"{split_name}.parquet"
            backup_path = Path(cfg["output"]["dir"]) / f"{split_name}_{backup_suffix}.parquet"
            if backup_path.exists():
                shutil.copy2(str(backup_path), str(orig_path))
                backup_path.unlink()
        sys.exit(1)

    run_calibrate()

    if args.shadow:
        copy_to_shadow()

    total_elapsed = datetime.now(timezone.utc) - start_time
    logger.info("Retrain pipeline complete in %s", total_elapsed)

    new_total = len(new_train) + len(new_val) + len(new_test)
    old_total = len(existing_train) + len(existing_val) + len(existing_test)
    logger.info("Dataset grew: %d -> %d (+%d examples)", old_total, new_total, new_total - old_total)

    summary = {
        "pipeline": "retrain",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": total_elapsed.total_seconds(),
        "old_total": old_total,
        "new_total": new_total,
        "new_examples": new_total - old_total,
        "shadow_copied": args.shadow,
    }
    with open(MODEL_DIR / "retrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", MODEL_DIR / "retrain_summary.json")


if __name__ == "__main__":
    main()
