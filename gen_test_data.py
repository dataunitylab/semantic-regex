import pandas as pd


data = ["['a', 'b', 'c']", "['1', '2', '3']"]
df = pd.DataFrame(data, columns=['values'])
df.to_parquet('test/test_values.parquet', index=True)

df = pd.DataFrame(data * 100, columns=['values'])
df.to_parquet('test/train_values.parquet', index=True)

labels = ["alpha", "numeric"]
df = pd.DataFrame(labels, columns=['type'])
df.to_parquet('test/test_labels.parquet', index=True)

df = pd.DataFrame(labels * 100, columns=['type'])
df.to_parquet('test/train_labels.parquet', index=True)
