import pandas as pd
import numpy as np

df = pd.read_csv("adult_subset_raw.csv")

def create_bins(column, num_bins):
    result, bins = pd.qcut(column, q=num_bins, duplicates='drop', retbins=True, labels=None)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    return pd.cut(column, bins=bins, labels=labels, include_lowest=True)

df['age'] = create_bins(df['age'], 5)  
df['education-num'] = create_bins(df['education-num'], 5) 
df['capital-gain'] = create_bins(df['capital-gain'], 4) 
df['hours-per-week'] = create_bins(df['hours-per-week'], 5) 
df['income'] = df['income'].apply(lambda x: '>=50k' if x == 1 else '<50k')
df = df.drop(columns=['capital-gain'])

output_path = "adult_subset.csv"
df.to_csv(output_path, index=False)

unique_values_counts = {col: df[col].value_counts() for col in df.columns}
print(unique_values_counts)