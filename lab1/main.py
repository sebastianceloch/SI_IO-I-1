import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
info = np.loadtxt('_info-data-discrete.txt', dtype="str")
diabetes = np.loadtxt('australian.txt')
diabetes_type = np.loadtxt('australian-type.txt', dtype="str")

df = pd.read_csv("australian.txt", sep=" ", header=None)
print(df)


columns = [
"a1s",
"a2n",
"a3n",
"a4s",
"a5s",
"a6s",
"a7n",
"a8s",
"a9s",
"a10n",
"a11s",
"a12s",
"a13n",
"a14n",
"yn"]
df.columns = columns
print(df.describe())
print(df.dtypes)
for x in columns:
    print(x, pd.unique(df[x]))

df2 = df.loc[:, df.columns != "yn"]
for x in columns[:-1]:
    random_rows = df2.sample(frac=0.1).index
    df2.loc[random_rows, x] = '?'
print(df2)

for x in columns[:-1]:
    count = len(df2[df2[x] == "?"])
    print(f'wartości ze znakiem zapytania w {x}: {count}')

df2.replace(to_replace='?', value=np.nan, inplace=True)
df2.apply(lambda x: x.fillna(x.mean()),axis=0)

print(df2)


for x in columns[:-1]:
    count = len(df2[df2[x] == "?"])
    print(f'wartości ze znakiem zapytania w {x}: {count}')

scaler = MinMaxScaler(feature_range=(-1, 1))
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_normalized)

scaler = MinMaxScaler(feature_range=(0, 1))
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_normalized)

normalized_df=(df-df.min())/(df.max()-df.min()) * 10
print(normalized_df)

scaler = StandardScaler()
df['a10n_std'] = scaler.fit_transform(df[['a10n']])
print(df['a10n_std'].describe())
df['a10n_std_2'] = (df['a10n'] - df['a10n'].mean())/df['a10n'].std()
print(df['a10n_std_2'].describe())

new_df = pd.read_csv("Churn_Modelling.csv")
print(new_df.head())

dummy_df = pd.get_dummies(new_df['Geography'])
print(dummy_df)

new_df = pd.concat([new_df, dummy_df], axis=1)
print(new_df)

new_df = new_df.drop("France", axis=1)
print(new_df)






