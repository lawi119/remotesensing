import pandas as pd
import os
import shutil

file = 'results.csv'
classes = list(set(file['target'].tolist()))
df = pd.read_csv(file)

for class in classes:
	tp = df[((df['target']==class)&(df['prediction']==class))].index()
	fp = df[((df['target']!=class)&(df['pred']==class))].index()
	tn = df[((df['target']!=class)&(df['pred']!=class))].index()
	fn = df[((df['target']==class)&(df['pred']!=class))].index()
	
	cats = [tp,fp,tn,fn]
	os.mkdir(output_folder, class)
	for cat in cats:
		os.mkdir(output_folder, class, cat)
		for item in cat:
			shutil.copy(df.iloc[item]['image'], os.path.join(output_folder, class, cat))
