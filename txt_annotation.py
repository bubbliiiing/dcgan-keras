import os
from os import getcwd
wd = getcwd()

datasets_path = "datasets/"
photos_names = os.listdir(datasets_path)
photos_names = sorted(photos_names)

list_file = open('train_lines.txt', 'w')
for photo_name in photos_names:
    if photo_name.endswith(".md"):
        continue
    list_file.write('%s/%s'%(wd, os.path.join(datasets_path, photo_name)))
    list_file.write('\n')
list_file.close()

