import pandas as pd
df = pd.read_csv("./data/Train_CSV_Balanced/train_LDM_balanced.csv")
print(df.head())
print(df['text'].value_counts())
