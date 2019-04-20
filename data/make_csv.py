import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

data_dir = 'data/ms-celebs/data/images'

classes = os.listdir(data_dir)

df = pd.DataFrame(columns=['image_path', 'mask_path', 'label'])
for c in tqdm(classes):
    image_dir = os.path.join(data_dir, c)
    mask_dir = os.path.join(data_dir.replace('images','masks'), c)
    if not os.path.isdir(image_dir):
        continue
    for i in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, i)
        mask_path = os.path.join(mask_dir, i.replace('.png','.xml'))
        if not os.path.exists(mask_path):
            mask_path = 'none'
        df = df.append({'image_path': image_path, 'mask_path': mask_path, 'label': c}, ignore_index=True)

dict((v,k) for k,v in enumerate(classes))


test_size = len(df)//5
df.label.value_counts()
df.sample(frac=0.2, random_state=1234).label.value_counts()
df = df.sample(frac=1, random_state=1234)
test_df = df.iloc[-test_size:]
train_df = df.iloc[:-test_size].reset_index(drop=True)


test_df = test_df.reset_index(drop=True)[['image_path','label']]

train_df.to_csv('data/ms-celebs/train.csv',index=False)
test_df.to_csv('data/ms-celebs/test.csv',index=False)

df.to_csv(os.path.join('data/ms-celebs/data.csv'), index=None)
