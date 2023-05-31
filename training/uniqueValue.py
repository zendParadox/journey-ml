import pandas as pd

# data = pd.read_csv('datasets/dataset1.csv')
data = pd.read_csv('datasets/asde.csv')
unique_values = data['disability_type'].unique()  # Gantilah 'column_name' dengan nama kolom yang ingin Anda cari nilai uniknya

print(unique_values)