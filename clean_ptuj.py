import pandas as pd

df = pd.read_csv("data/preprocessed/temp/PTUJ.csv")

print(df)
# Drop columns that are all empty or all NaN
df = df.dropna(axis=1, how='all')
df.to_csv("PTUJ_2.csv", index=False)