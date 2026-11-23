"""
End-to-end smoke test (CPU or GPU) for the IndoBERT clickbait pipeline.

Runs the REAL code path -- DataManager -> ModelTrainer (1 epoch) -> checkpoint
save/load -> PerturbationEngine -> EvaluationEngine -- on a tiny stratified
sample of the real dataset, in a throw-away temp directory.  It needs the real
torch/transformers install and the cached IndoBERT weights, but no GPU.

Usage (from ths_test/):
    python src/smoke_test.py                  # auto device, 24 rows/domain
    python src/smoke_test.py --device cpu --rows 24 --keep
    python src/smoke_test.py --device cuda    # also validates AMP / pin_memory path

Exit code 0 = every check passed; non-zero = first failure (traceback printed).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DOMAIN_FILES = {
    "Technology": "technology.csv", "Politics": "politic.csv", "Health": "health.csv",
    "Sport": "sport.csv", "Education": "education.csv",
}


def check(cond: bool, msg: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise AssertionError(msg)


def build_tiny_dataset(src_dir: str, dst_dir: str, rows: int, seed: int) -> None:
    """Class-balanced sample of `rows` per domain, same ';' schema as the real CSVs."""
    os.makedirs(dst_dir, exist_ok=True)
    half = rows // 2
    for fname in DOMAIN_FILES.values():
        df = pd.read_csv(os.path.join(src_dir, fname), sep=";")
        parts = [g.sample(n=half, random_state=seed) for _, g in df.groupby("Label")]
        pd.concat(parts).sample(frac=1, random_state=seed).to_csv(
            os.path.join(dst_dir, fname), sep=";", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--rows", type=int, default=24, help="rows per domain (even, >=20)")
    ap.add_argument("--model", default="base", choices=["base", "large", "lite"])
    ap.add_argument("--dataset-dir", default=os.path.join(HERE, "..", "dataset", "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep", action="store_true", help="keep the temp dir for inspection")
    args = ap.parse_args()

    import torch
    from config import config
    from train_pipeline import TrainingPipeline
    from evaluate_pipeline import EvaluationPipeline

    dataset_dir = os.path.abspath(args.dataset_dir)
    thesaurus = os.path.join(dataset_dir, "dict.json")
    check(os.path.isfile(thesaurus), f"thesaurus found: {thesaurus}")

    tmp = tempfile.mkdtemp(prefix="smoke_")
    print(f"\n=== SMOKE TEST  device={args.device}  rows/domain={args.rows}  tmp={tmp}")
    t0 = time.time()
    try:
        tiny = os.path.join(tmp, "dataset")
        out = os.path.join(tmp, "output")
        ckpt = os.path.join(tmp, "checkpoints")
        res = os.path.join(tmp, "evaluation_results")
        pert = os.path.join(tmp, "perturbation")
        build_tiny_dataset(dataset_dir, tiny, args.rows, args.seed)

        # Minimal hyper-parameters: 1 epoch, small batch.  (restored on exit via process end)
        config.training.NUM_EPOCHS = 1
        config.training.BATCH_SIZE = 8
        config.system.NUM_WORKERS = 2  # exercised only when device is CUDA

        # ── 1. TRAIN: DataManager -> tokenisation -> train/val -> checkpoint ──
        print("\n[1/4] Training pipeline (1 epoch, 5 domains)")
        tp = TrainingPipeline(dataset_dir=tiny, output_dir=out, checkpoint_dir=ckpt,
                              device=args.device, random_seed=args.seed)
        train_res = tp.run(models=[args.model], perform_grid_search=False)
        domains = train_res["data_info"]["domains"]
        check(len(domains) == 5, f"5 domains loaded: {domains}")
        for d in domains:
            info = train_res["models"][args.model][d]
            ck = os.path.join(ckpt, args.model, d, "best_model.pt")
            check(os.path.isfile(ck) and os.path.getsize(ck) > 1_000_000,
                  f"{d}: best_model.pt written ({os.path.getsize(ck)/1e6:.0f} MB)")
            check(not os.path.exists(ck + ".tmp"), f"{d}: no stray .tmp checkpoint")
            check(0.0 <= info["best_macro_f1"] <= 1.0, f"{d}: val macro-F1 in [0,1]")
        check(os.path.isfile(os.path.join(out, "training_summary.json")), "training_summary.json saved")
        check(all(os.path.isdir(os.path.join(out, "data_splits", d.lower())) for d in domains),
              "data_splits/<domain>/ exported")

        # ── 2. CHECKPOINT ROUND-TRIP + SHAPES ──
        print("\n[2/4] Checkpoint round-trip, tensor shapes, determinism")
        from model_trainer import ModelTrainer
        full = getattr(config.model, {"base": "INDOBERT_BASE", "large": "INDOBERT_LARGE",
                                      "lite": "INDOBERT_LITE"}[args.model])
        trainer = ModelTrainer(model_name=full, max_length=config.model.MAX_SEQUENCE_LENGTH,
                               device=args.device, random_seed=args.seed)
        trainer.load_checkpoint(os.path.join(ckpt, args.model, domains[0], "best_model.pt"))
        texts = ["Ini judul berita biasa", "Kamu Tidak Akan Percaya Apa yang Terjadi Selanjutnya!", "Tes"]
        preds, probs = trainer.predict(texts)
        check(preds.shape == (3,), f"predict preds shape {preds.shape} == (3,)")
        check(probs.shape == (3, 2), f"predict probs shape {probs.shape} == (3, 2)")
        check(np.allclose(probs.sum(axis=1), 1.0, atol=1e-4), "softmax rows sum to 1")
        check(np.isfinite(probs).all(), "no NaN/Inf in probabilities")
        preds2, probs2 = trainer.predict(texts)
        check(np.allclose(probs, probs2, atol=1e-5), "inference deterministic across two calls")
        batch = trainer.tokenizer(texts, max_length=config.model.MAX_SEQUENCE_LENGTH,
                                  padding="max_length", truncation=True, return_tensors="pt")
        check(tuple(batch["input_ids"].shape) == (3, config.model.MAX_SEQUENCE_LENGTH),
              f"tokenised batch shape {tuple(batch['input_ids'].shape)}")
        print(f"      device={trainer.device}  amp_dtype={trainer.amp_dtype}  "
              f"workers={trainer.num_workers}  scaler={'on' if trainer.scaler else 'off'}")
        trainer.release_memory()
        check(trainer.model is None, "release_memory() dropped the model")

        # ── 3. EVALUATE: in/cross-domain + perturbation (semantic) ──
        print("\n[3/4] Evaluation pipeline (incl. perturbation; statistics skipped)")
        ep = EvaluationPipeline(checkpoint_dir=ckpt, dataset_dir=tiny, output_dir=res,
                                train_output_dir=out, device=args.device, random_seed=args.seed,
                                thesaurus_path=thesaurus, perturbation_output_dir=pert)
        eres = ep.run(models=[args.model], skip_perturbation=False, skip_statistics=True)
        check(eres["status"] == "success", "evaluate pipeline status == success")
        r = eres["results"][args.model]
        check(len(r["in_domain"]) == 5, "5 in-domain results")
        check(len(r["cross_domain"]) == 25, "25 cross-domain (5x5) results")
        check(set(r["perturbation"].keys()) == set(domains), "perturbation results for all domains")
        for d in domains:
            check(all(l in r["perturbation"][d]["semantic"] for l in ("low", "medium", "high")),
                  f"{d}: low/medium/high perturbation scored")

        # ── 4. OUTPUT FILES ──
        print("\n[4/4] Output directory contents")
        eval_json = os.path.join(res, args.model, "complete_evaluation.json")
        check(os.path.isfile(eval_json), "complete_evaluation.json saved")
        with open(eval_json) as f:
            loaded = json.load(f)
        check(all("->" in k for k in loaded["cross_domain"]), "cross_domain keys serialised as 'A->B'")
        check(os.path.isfile(os.path.join(res, "evaluation_summary.json")), "evaluation_summary.json saved")
        n_csv = sum(f.endswith(".csv") for _, _, fs in os.walk(os.path.join(pert, args.model)) for f in fs)
        check(n_csv >= 15, f"perturbed CSVs written ({n_csv} files)")

        print(f"\n=== SMOKE TEST PASSED in {time.time() - t0:.0f}s")
        return 0
    finally:
        if args.keep:
            print(f"(kept) {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
