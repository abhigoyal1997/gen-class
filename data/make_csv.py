import os
import pandas as pd
import sys
from tqdm import tqdm

seed = int(sys.argv[1])

data_dir = 'data/ms-celebs/data2/images'

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

test_size = len(df)//5
print()
print(df.sample(frac=0.2, random_state=seed).label.value_counts())
df = df.sample(frac=1, random_state=seed)
test_df = df.iloc[-test_size:]
train_df = df.iloc[:-test_size].reset_index(drop=True)


test_df = test_df.reset_index(drop=True)[['image_path','label']]

train_df.to_csv(f'data/ms-celebs/data2/train{seed}.csv',index=False)
test_df.to_csv(f'data/ms-celebs/data2/test{seed}.csv',index=False)
