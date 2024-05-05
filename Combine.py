import pandas as pd
import os

excel_dir = "Dataset\\"
excel_files = [f for f in os.listdir(excel_dir) if f.endswith('.xlsx')]

dfs = []
for excel_file in excel_files:
    file_path = os.path.join(excel_dir, excel_file)
    df = pd.read_excel(file_path)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True, sort=False)
output_path = "Combined.xlsx"
combined_df.to_excel(output_path, index=False)