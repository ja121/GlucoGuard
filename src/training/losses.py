import torch
import torch.nn as nn
import torch.nn.functional as F

class GlucosePredictionLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha # Weight for glucose MSE
        self.beta = beta # Weight for risk classification
        self.gamma = gamma # Weight for uncertainty

    def forward(self, predictions, targets):
        # Glucose prediction loss (MSE with Clarke Error Grid penalty)
        glucose_loss = self._clarke_error_grid_loss(
            predictions['glucose'],
            targets['glucose']
        )

        # Risk classification loss (Focal loss for imbalanced classes)
        risk_loss = self._focal_loss(
            predictions['risk'],
            targets['risk']
        )

        # Uncertainty loss (negative log-likelihood)
        uncertainty_loss = self._uncertainty_loss(
            predictions['glucose'],
            targets['glucose'],
            predictions['uncertainty']
        )

        total_loss = (self.alpha * glucose_loss +
                     self.beta * risk_loss +
                     self.gamma * uncertainty_loss)

        return {
            'total': total_loss,
            'glucose': glucose_loss,
            'risk': risk_loss,
            'uncertainty': uncertainty_loss
        }

    def _clarke_error_grid_loss(self, pred, target):
        """Penalize predictions based on clinical significance"""
        # Remove nan targets
        valid_indices = ~torch.isnan(target)
        pred = pred[valid_indices]
        target = target[valid_indices]

        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        mse = F.mse_loss(pred, target)

        # Clarke Error Grid zones
        error = pred - target

        # Zone A (clinically accurate) - no penalty
        zone_a = (torch.abs(error) <= 20) | \
                 ((target >= 70) & (pred >= 70) & (torch.abs(error / target) <= 0.2))

        # Zone B (benign errors) - small penalty
        zone_b = ~zone_a & (torch.abs(error) <= 40)

        # Zone C-E (dangerous errors) - high penalty
        zone_cde = ~zone_a & ~zone_b

        # Weighted loss
        weights = torch.ones_like(error)
        weights[zone_b] = 1.5
        weights[zone_cde] = 3.0

        weighted_mse = (weights * (error ** 2)).mean()

        return weighted_mse

    def _focal_loss(self, pred, target, gamma=2.0, alpha=0.25):
        """Focal loss for imbalanced classification"""
        # Remove nan targets
        valid_indices = ~torch.isnan(target)
        pred = pred[valid_indices]
        target = target[valid_indices]

        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** gamma

        if alpha is not None:
            alpha_t = torch.where(target == 1, alpha, 1 - alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
        return loss.mean()

    def _uncertainty_loss(self, pred, target, uncertainty):
        """Negative log-likelihood for uncertainty"""
        # Remove nan targets
        valid_indices = ~torch.isnan(target)
        pred = pred[valid_indices]
        target = target[valid_indices]
        uncertainty = uncertainty[valid_indices]

        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        return (0.5 * torch.log(uncertainty) + 0.5 * ((pred - target) ** 2) / uncertainty).mean()
