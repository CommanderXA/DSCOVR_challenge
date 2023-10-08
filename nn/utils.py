import torch


@torch.no_grad
def evaluate_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, tolerance: float
) -> float:
    diff = torch.abs(logits - targets)
    tolerated = diff <= tolerance
    correct = torch.sum(tolerated, dim=0).item()
    sample = targets.size(0)
    accuracy = 100 * (correct / sample)
    return accuracy
