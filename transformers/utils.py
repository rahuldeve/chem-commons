from dataclasses import dataclass

@dataclass
class TrainArgs:
    B: int = 8
    n_candidates: int = 16
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-6
    gradient_accumulation_steps: int = 2
    epochs: int = 30
    lr: float = 2e-5
    warmup_proportion: float = 0.2
    seed: int = 42
    clip: float = 1.0
