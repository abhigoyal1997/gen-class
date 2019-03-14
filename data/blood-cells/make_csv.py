import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

data_dir = '/data1/abhinav/blood-cells/dataset2-master/mixed/images'

classes = os.listdir(data_dir)

df = pd.DataFrame(columns=['image_path', 'mask_path', 'label'])
for c in tqdm(classes):
    image_dir = os.path.join(data_dir, c)
    mask_dir = os.path.join(data_dir.replace('images','masks'), c)
    if not os.path.isdir(image_dir):
        continue
    for i in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, i)
        mask_path = os.path.join(mask_dir, i)
        if not os.path.exists(mask_path):
            mask_path = 'none'
        df = df.append({'image_path': image_path, 'mask_path': mask_path, 'label': c}, ignore_index=True)


test_size = len(df)//5
df.label.value_counts()
df[df.mask_path == 'none']
df[df.mask_path == 'none'].sample(frac=0.2, random_state=1234).label.value_counts()
df_no_mask = df[df.mask_path == 'none'].sample(frac=1, random_state=1234)
test_df = df_no_mask.iloc[-test_size:]
train_df = df[df.mask_path != 'none']

train_df = pd.concat([train_df, df_no_mask.iloc[:-test_size]]).reset_index(drop=True)

test_df = test_df.reset_index(drop=True)[['image_path','label']]

train_df.to_csv('data/mixed_train.csv',index=False)
test_df.to_csv('data/mixed_test.csv',index=False)

df.to_csv(os.path.join('data/mixed.csv'), index=None)
