import pandas as pd
from src.preprocessing import build_preprocessor

def test_preprocessor_build():
    preprocessor = build_preprocessor(["age"], ["gender"])
    assert preprocessor is not None
