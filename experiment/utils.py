import ray

@ray.remote
class RayExperimentTracker:
    def __init__(self) -> None:
        self.progress:int = 0

    def get_progress(self) -> int:
        return self.progress
    
    def update(self) -> None:
        self.progress = self.progress + 1