from dataclasses import dataclass

@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    model_path: str = "models/credit_model.pkl"
    threshold: float = 0.5
