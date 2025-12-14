
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import convert_datetime


import pandas as pd
from src.data_processing import convert_datetime

def test_convert_datetime():
    df = pd.DataFrame({
        "TransactionStartTime": ["2024-01-01", "2024-01-02"]
    })

    result = convert_datetime(df, "TransactionStartTime")

    assert pd.api.types.is_datetime64_any_dtype(result["TransactionStartTime"])
    assert str(result["TransactionStartTime"].dtype).startswith("datetime64")
