import torch
import torch.nn as nn

class MaskedLoss(nn.Module):
    def __init__(self):
        """
        Initializes the MaskedLoss.

        Args:
            base_loss_fn: A PyTorch loss function (e.g., nn.MSELoss, nn.L1Loss) that will be applied to the unmasked values.
        """
        super(MaskedLoss, self).__init__()
        self.base_loss_fn = nn.MSELoss()

    def forward(self, predictions, targets, mask=None):
        """
        Computes the masked loss.

        Args:
            predictions: The predicted output from the model (x = 20% masked).
            targets: The ground truth with 10% masked (y = 10% masked).
            mask: A binary mask tensor indicating which values were masked (1 if masked, 0 otherwise).

        Returns:
            The computed masked loss.
        """

        if mask is None:
            # If no mask is provided, treat all values as unmasked
            mask = torch.zeros_like(targets, dtype=torch.bool)

        # Invert the mask so that 1 corresponds to unmasked values
        unmasked = torch.logical_not(mask)

        # Compute the loss only for the unmasked values
        loss = self.base_loss_fn(predictions * unmasked, targets * unmasked)
        
        # Normalize by the number of unmasked elements to avoid scale issues
        normalization = unmasked.sum()
        if normalization == 0:
            return torch.tensor(0.0, requires_grad=True)  # Avoid division by zero
        loss = loss / normalization

        return loss

