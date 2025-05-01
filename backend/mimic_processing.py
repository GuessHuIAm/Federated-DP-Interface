import pandas as pd, numpy as np

RAW = "./datasets/mimic_iv_hypotensive_cut2.csv"
DROP = ['vaso(amount)', 'bolus(amount)',
        'any_treatment(binary)', 'vaso(binary)', 'bolus(binary)']

df = (pd.read_csv(RAW, sep=',')
        .sort_values(['stay_id','time']))

df['action'] = 2*df['bolus(binary)'] + df['vaso(binary)']

df = df.drop(columns=DROP)

df = (df.groupby('stay_id').ffill()
        .fillna(df.mean(numeric_only=True)))

DROP_EXTRA = ["time"]
df = df.drop(columns=DROP_EXTRA)

df.sample(frac=1.0, random_state=0).reset_index(drop=True)\
  .to_csv("./datasets/MIMIC_hypotension_FL.csv", index=False)
