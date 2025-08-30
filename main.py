import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data_processing.cgm_dataset import AdvancedCGMDataset, CGMConfig
from src.models.predictor import HierarchicalGlucosePredictor
from src.training.lightning_module import GlucoseLightningModule

def main(args):
    """Main training script"""

    # Configuration
    config = CGMConfig()

    # Dataset and DataLoader
    # Note: This assumes the 'Brown2019' dataset is available via the adapter.
    # In a real scenario, you would need to download and place the data correctly.
    try:
        train_dataset = AdvancedCGMDataset(dataset_name='Brown2019', config=config, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        # Using a subset of the same data for validation for demonstration purposes
        val_dataset = AdvancedCGMDataset(dataset_name='Brown2019', config=config, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        print("Please make sure the dataset is available. The adapter currently requires manual download.")
        return

    # Model
    # The number of features needs to be calculated from the dataset
    # This is a placeholder value. A more robust implementation would calculate this.
    num_features = 50
    model = HierarchicalGlucosePredictor(config=config, n_features=num_features)

    # Lightning Module
    lightning_module = GlucoseLightningModule(model=model, config=config)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices='auto',
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val/loss', patience=5),
            pl.callbacks.ModelCheckpoint(monitor='val/loss', save_top_k=1, mode='min')
        ]
    )

    # Start training
    trainer.fit(lightning_module, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    main(args)
