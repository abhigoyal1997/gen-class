import os
import pandas as pd

data_dir = '/data1/abhinav/blood-cells/dataset2-master/complete/test_simple'

classes = os.listdir(data_dir)

df = pd.DataFrame(columns=['image_path', 'label'])
for c in classes:
    image_dir = os.path.join(data_dir, c)
    if not os.path.isdir(image_dir):
        continue
    for i in os.listdir(image_dir):
        image_path = os.path.join(image_dir, i)

        df = df.append({'image_path': image_path, 'label': c}, ignore_index=True)

df.to_csv(os.path.join('test_data2.csv'), index=None)
