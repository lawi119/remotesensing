import pandas as pd

df = pd.read_csv('/data/datasets/esc50/ESC-50-master/meta/esc50.csv')
df['filename'] = [x.split('/')[-1].split('.wav')[0]+'.png' for x in df['filename']]
df.to_csv('/data/datasets/esc50/esc50_processed/meta/esc50.csv', index=None)
df_train = df[df['fold'].isin([1,2,3,4])]
df_test = df[df['fold'].isin([5])]
df_train.to_csv('/data/datasets/esc50/esc50_processed/meta/esc50_train.csv', index=None)
df_test.to_csv('/data/datasets/esc50/esc50_processed/meta/esc50_test.csv', index=None)
