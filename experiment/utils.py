import ray
import numpy as np
import random

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

@ray.remote
class RayExperimentTracker:
    def __init__(self) -> None:
        self.progress:int = 0

    def get_progress(self) -> int:
        return self.progress
    
    def update(self) -> None:
        self.progress = self.progress + 1