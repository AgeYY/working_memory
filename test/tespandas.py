import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df['c'] = [7, 8, 9]

print(df.columns)
