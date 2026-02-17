from dataclasses import dataclass

@dataclass
class ModelConfig:
   target_column = "FraudResult"# <-- Replace with actual column name
   test_size: float = 0.2
   random_state: int = 42
   model_path: str = "models/credit_model.pkl"


