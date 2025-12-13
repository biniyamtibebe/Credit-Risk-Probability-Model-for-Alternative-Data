import shutil

# Specify the source file path and the destination path
source = r'C:\Users\hp\Downloads\Xente_Variable_Definitions.csv'  # Update this to your actual file
destination = r'C:\Users\hp\Pictures\Bati Bank\Credit-Risk-Probability-Model-for-Alternative-Data\data\raw\Xente_Variable_Definitions.csv'

# Copying the file
shutil.copy(source, destination)

print("File copied successfully!")