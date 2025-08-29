import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import torch.nn.functional as F

from src.models.predictor import HierarchicalGlucosePredictor
from src.data_processing.cgm_dataset import CGMConfig
from src.training.losses import GlucosePredictionLoss

class GlucoseLightningModule(pl.LightningModule):
    def __init__(self, model: HierarchicalGlucosePredictor, config: CGMConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = GlucosePredictionLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        predictions = self(x)

        losses = self.loss_fn(predictions, y)

        # Log metrics
        self.log('train/loss', losses['total'], prog_bar=True)
        self.log('train/glucose_mae', F.l1_loss(predictions['glucose'], y['glucose']))

        # Calculate and log clinical metrics
        clinical_metrics = self._calculate_clinical_metrics(predictions, y)
        for name, value in clinical_metrics.items():
            self.log(f'train/{name}', value)

        return losses['total']

    def validation_step(self, batch, batch_idx):
        x, y, metadata = batch
        predictions = self(x)

        losses = self.loss_fn(predictions, y)

        # Calculate comprehensive metrics
        metrics = {
            'val/loss': losses['total'],
            'val/glucose_mae': F.l1_loss(predictions['glucose'], y['glucose']),
            'val/glucose_rmse': torch.sqrt(F.mse_loss(predictions['glucose'], y['glucose'])),
        }

        # Clinical metrics
        clinical_metrics = self._calculate_clinical_metrics(predictions, y)
        for name, value in clinical_metrics.items():
            metrics[f'val/{name}'] = value

        self.log_dict(metrics, prog_bar=True)

        return metrics

    def _calculate_clinical_metrics(self, pred, target):
        """Calculate clinically relevant metrics"""
        # Ensure target is a dictionary
        if not isinstance(target, dict) or 'glucose' not in target:
            return {}

        glucose_pred = pred['glucose'][:, 0]  # 30-min prediction
        glucose_true = target['glucose'][:, 0]

        valid_indices = ~torch.isnan(glucose_true)
        glucose_pred = glucose_pred[valid_indices]
        glucose_true = glucose_true[valid_indices]

        if glucose_true.numel() == 0:
            return {}

        # MARD (Mean Absolute Relative Difference)
        mard = (torch.abs(glucose_pred - glucose_true) / glucose_true * 100).mean()

        # Surveillance Error Grid Analysis
        seg_scores = self._surveillance_error_grid(glucose_pred, glucose_true)

        # Hypoglycemia detection metrics
        hypo_pred = pred['risk'][:, 0] > 0.5
        hypo_true = target['risk'][:, 0] > 0.5

        valid_risk_indices = ~torch.isnan(hypo_true)
        hypo_pred = hypo_pred[valid_risk_indices]
        hypo_true = hypo_true[valid_risk_indices]

        if hypo_true.numel() == 0:
            sensitivity = torch.tensor(0.0)
            precision = torch.tensor(0.0)
            f1 = torch.tensor(0.0)
        else:
            tp = ((hypo_pred == 1) & (hypo_true == 1)).sum().float()
            fp = ((hypo_pred == 1) & (hypo_true == 0)).sum().float()
            fn = ((hypo_pred == 0) & (hypo_true == 1)).sum().float()

            sensitivity = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)

        return {
            'mard': mard,
            'seg_a': seg_scores['A'],
            'seg_b': seg_scores['B'],
            'hypo_sensitivity': sensitivity,
            'hypo_f1': f1
        }

    def _surveillance_error_grid(self, pred, target):
        """Placeholder for Surveillance Error Grid Analysis"""
        # This is a complex clinical metric. Returning dummy values for now.
        return {'A': 1.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0}

    def configure_optimizers(self):
        # AdamW with weight decay
        optimizer = AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
