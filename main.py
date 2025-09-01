import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data_processing.cgm_dataset import AdvancedCGMDataset, CGMConfig
from src.models.predictor import HierarchicalGlucosePredictor
from src.training.lightning_module import GlucoseLightningModule
from src.data_processing.adapter import AwesomeCGMAdapter

def main(args):
    """Main training script"""

    # 1. Configuration from args
    config = CGMConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        d_model_cgm=args.d_model_cgm,
        d_model_wearable=args.d_model_wearable,
        d_model_fusion=args.d_model_fusion,
        n_attention_heads=args.n_attention_heads,
        n_attention_layers=args.n_attention_layers,
    )

    # 2. Load and split data
    print(f"Loading dataset: {args.dataset_name}...")
    try:
        adapter = AwesomeCGMAdapter()
        raw_data = adapter.load_dataset(args.dataset_name)
        prepared_data = adapter.prepare_for_model(raw_data)
    except FileNotFoundError:
        print(f"Error: Dataset file for '{args.dataset_name}' not found.")
        print("Please make sure the data is downloaded and placed in the 'cgm_data' directory.")
        return

    subject_ids = prepared_data['subject_id'].unique()
    train_ids, val_ids = train_test_split(subject_ids, test_size=0.2, random_state=42)

    train_df = prepared_data[prepared_data['subject_id'].isin(train_ids)]
    val_df = prepared_data[prepared_data['subject_id'].isin(val_ids)]
    print(f"Data loaded. Training on {len(train_ids)} subjects, validating on {len(val_ids)} subjects.")

    # 3. Create Datasets and DataLoaders
    train_dataset = AdvancedCGMDataset(config=config, dataframe=train_df, mode='train')
    val_dataset = AdvancedCGMDataset(config=config, dataframe=val_df, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Get feature dimensions from the dataset
    cgm_feats, wearable_feats, _, _ = train_dataset[0]
    n_cgm_features = cgm_feats.shape[1]
    n_wearable_features = wearable_feats.shape[1]
    print(f"Detected {n_cgm_features} CGM features and {n_wearable_features} wearable features.")

    # 5. Model
    model = HierarchicalGlucosePredictor(
        config=config,
        n_cgm_features=n_cgm_features,
        n_wearable_features=n_wearable_features
    )

    # 6. Lightning Module
    lightning_module = GlucoseLightningModule(model=model, config=config)

    # 7. Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices='auto',
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val/loss', patience=10, mode='min'),
            pl.callbacks.ModelCheckpoint(monitor='val/loss', save_top_k=1, mode='min', dirpath='checkpoints/', filename=f'{args.dataset_name}-best-model')
        ],
        logger=pl.loggers.TensorBoardLogger("training_logs/", name=args.dataset_name)
    )

    # 8. Start training
    print("Starting training...")
    trainer.fit(lightning_module, train_loader, val_loader)
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the GlucoGuard model.")

    # --- Data and Training arguments ---
    parser.add_argument('--dataset_name', type=str, default='Brown2019', help='Name of the dataset to train on.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)

    # --- Model Hyperparameters ---
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--d_model_cgm', type=int, default=128)
    parser.add_argument('--d_model_wearable', type=int, default=64)
    parser.add_argument('--d_model_fusion', type=int, default=128)
    parser.add_argument('--n_attention_heads', type=int, default=8)
    parser.add_argument('--n_attention_layers', type=int, default=3)

    args = parser.parse_args()
    main(args)
