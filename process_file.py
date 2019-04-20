import sys
import os
from base64 import decodebytes

in_file = sys.argv[1]
out_folder = sys.argv[2]

with open(sys.argv[1],'rb') as f:
    data = f.readlines()

num_images = {}

celebs = {}

for d in data:
    d = d.split(b'\t')
    entity = d[0].decode('utf-8')

    folder = os.path.join(out_folder,entity)
    celebs[entity] = d[1].decode('utf-8')

    if not os.path.exists(folder):
        os.makedirs(folder)

    if entity in num_images:
        num_images[entity] += 1
    else:
        num_images[entity] = 1

    img_data = decodebytes(d[-1])

    with open(os.path.join(folder, str(num_images[entity]) + '.png'),'wb') as f:
        f.write(img_data)

with open(os.path.join(out_folder, 'celeb_list'),'w') as f:
    f.write('\n'.join([f'{k}\t{v}' for k,v in celebs.items()]))
