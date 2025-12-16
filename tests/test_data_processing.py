# Unit Testing (tests/test_data_processing.py)

from src.data_processing import create_aggregates
import pandas as pd

def test_create_aggregates():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "Amount": [100, 200, 50],
        "TransactionId": [1, 2, 3]
    })

    result = create_aggregates(df)
    
    assert "total_amount" in result.columns
    assert result.loc[result.CustomerId == 1, "total_amount"].iloc[0] == 300