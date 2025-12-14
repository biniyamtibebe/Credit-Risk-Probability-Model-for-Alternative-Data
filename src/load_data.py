#IMPORT LIBRARIES


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")

# Enable full-width display
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1500)

# 2. LOAD DATA


# Place your downloaded data in: data/raw/xente_dataset.csv
data_path = r"C:\Users\hp\Pictures\Bati Bank\Credit-Risk-Probability-Model-for-Alternative-Data\data\raw\CreditRisk-data.csv"

df = pd.read_csv(data_path)

# View structure
df.head()
