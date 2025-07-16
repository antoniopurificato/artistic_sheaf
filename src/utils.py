def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across multiple libraries.
    
    Args:
        seed: Integer seed for random number generation
    """
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)