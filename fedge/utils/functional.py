import torch

def vectorize(param_list: list[torch.Tensor]) -> torch.Tensor:
    """Flatten and concatenate a list of Tensors into one vector."""
    return torch.cat([p.detach().flatten() for p in param_list], dim=0)
