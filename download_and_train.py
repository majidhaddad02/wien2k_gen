#!/usr/bin/env python3
"""One-shot script: download MP dataset, train GNN, save model.

Run from the forge project root:
    export MP_API_KEY="your_key_here"
    python3 download_and_train.py

The model will be saved to ~/.forge/models/kpoint_predictor.npz
and automatically used by forge when predicting k-point grids.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from forge.ml.data_pipeline import MPDatasetPipeline

OUTPUT_DIR = os.path.expanduser("~/.forge/mp_data")
N_STRUCTURES = 2000

if __name__ == "__main__":
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        print("ERROR: Set MP_API_KEY environment variable first.")
        print("  export MP_API_KEY='your_key_here'")
        sys.exit(1)

    print(f"Step 1/2: Downloading {N_STRUCTURES} structures from Materials Project...")
    pipeline = MPDatasetPipeline(api_key=api_key)
    meta = pipeline.build_dataset(
        n_structures=N_STRUCTURES,
        output_dir=OUTPUT_DIR,
    )
    print(f"  Downloaded {meta['n_total_structures']} structures")
    print(f"  Train: {meta['n_train']}, Val: {meta['n_val']}")

    print(f"\nStep 2/2: Training GNN model ({meta['n_train']} samples, 30 epochs)...")
    from forge.ml.gnn_kpoint_predictor import _train_from_mp_dataset

    model, ok = _train_from_mp_dataset()
    if ok:
        print("Training complete — model saved to ~/.forge/models/kpoint_predictor.npz")
    else:
        print("ERROR: Training failed — check logs above.")
        sys.exit(1)

    print("\nDone. Forge will now use the trained model automatically.")
