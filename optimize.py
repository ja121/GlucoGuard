import optuna
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project-specific imports
from src.data_processing.cgm_dataset import AdvancedCGMDataset, CGMConfig
from src.models.predictor import HierarchicalGlucosePredictor
from src.training.lightning_module import GlucoseLightningModule
from src.data_processing.adapter import AwesomeCGMAdapter
from optuna.integration import PyTorchLightningPruningCallback

import logging
import sys

# Add a logger for Optuna
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# --- Constants ---
DATASET_NAME = "Brown2019" # Default dataset for optimization
N_TRIALS = 25 # Number of optimization trials to run
N_EPOCHS = 10 # Number of epochs to train per trial
BATCH_SIZE = 64

def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna objective function.
    Trains a model with a set of hyperparameters and returns the validation loss.
    """
    # 1. Define hyperparameters to tune
    config = CGMConfig(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5),
        d_model_cgm=trial.suggest_categorical("d_model_cgm", [64, 128]),
        d_model_wearable=trial.suggest_categorical("d_model_wearable", [32, 64]),
        d_model_fusion=trial.suggest_categorical("d_model_fusion", [128, 256]),
        n_attention_heads=trial.suggest_categorical("n_attention_heads", [4, 8]),
        n_attention_layers=trial.suggest_int("n_attention_layers", 2, 4),
    )

    # 2. Load and split data
    adapter = AwesomeCGMAdapter()
    raw_data = adapter.load_dataset(DATASET_NAME)
    prepared_data = adapter.prepare_for_model(raw_data)

    subject_ids = prepared_data['subject_id'].unique()
    train_ids, val_ids = train_test_split(subject_ids, test_size=0.2, random_state=42)

    train_df = prepared_data[prepared_data['subject_id'].isin(train_ids)]
    val_df = prepared_data[prepared_data['subject_id'].isin(val_ids)]

    # 3. Create Datasets and DataLoaders
    train_dataset = AdvancedCGMDataset(config=config, dataframe=train_df, mode='train')
    val_dataset = AdvancedCGMDataset(config=config, dataframe=val_df, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Instantiate Model and LightningModule
    # We need to get the feature dimensions from the dataset
    cgm_feats, wearable_feats, _, _ = train_dataset[0]
    n_cgm_features = cgm_feats.shape[1]
    n_wearable_features = wearable_feats.shape[1]

    model = HierarchicalGlucosePredictor(
        config=config,
        n_cgm_features=n_cgm_features,
        n_wearable_features=n_wearable_features
    )
    lightning_module = GlucoseLightningModule(model=model, config=config)

    # 5. Create Trainer with Pruning Callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/loss")
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator='auto',
        callbacks=[pruning_callback],
        logger=pl.loggers.TensorBoardLogger("optuna_logs/", name=f"trial_{trial.number}"),
        enable_progress_bar=True # Set to False to reduce console spam
    )

    # 6. Run Fit
    trainer.fit(lightning_module, train_loader, val_loader)

    # 7. Return Metric
    # The metric is automatically reported by the callback, but we can also return it.
    return pruning_callback.best_score.item()


if __name__ == "__main__":
    # --- Study Creation ---
    # We use a TPE sampler for efficient search.
    # We use a median pruner to stop unpromising trials early.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    # --- Run Optimization ---
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Results ---
    print("\n\nOptimization finished.")
    print("Best trial:")
    best = study.best_trial
    print(f"  Value (Validation Loss): {best.value}")
    print("  Params: ")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
