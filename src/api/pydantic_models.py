## 13. FastAPI with Validation (`src/api/pydantic_models.py`)

from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    total_amount: float
    avg_amount: float
    txn_count: int
    amount_std: float

class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: float