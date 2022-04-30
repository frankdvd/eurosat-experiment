# import required libraries
import numpy as gfg
import matplotlib.image as img

import numpy as np
import sys
import os
import csv

root_folder = './data/2750'

classes_map = {
    'AnnualCrop': 0,
    'Forest':1,
    'HerbaceousVegetation':2,
    'Highway':3,
    'Industrial':4,
    'Pasture':5,
    'PermanentCrop':6,
    'Residential':7,
    'River':8,
    'SeaLake':9
}


#Useful function
def createFileList(root_dir, format='.jpg'):
    file_dict = {}
    print(root_dir)
    for root_root, root_dirs, root_files in os.walk(root_dir, topdown=False):
        for dir in root_dirs:
            print(dir)
            file_dict[classes_map[dir]] = []
            for root, dirs, files in os.walk(os.path.join(root_root, dir), topdown=False):
                for name in files:
                    if name.endswith(format):
                        fullName = os.path.join(root, name)
                        file_dict[classes_map[dir]].append(fullName)
    return file_dict

images = createFileList(root_folder)


csv_path = str('./data/csv_data.csv')  

csv_header = []
for i in range(64*64*3):
    csv_header.append('p_' + str(i))
csv_header.append('class')

# open the file in the write mode
csv_f = open(csv_path, 'w', encoding='UTF8', newline='')

# create the csv writer
csv_writer = csv.writer(csv_f)

# write a row to the csv file
csv_writer.writerow(csv_header)


result = []
for key in images:
    for file in images[key]:
        # read an image
        imageMat = img.imread(file)
        # reshape it from 3D matrice to 1D matrice
        imageMat_reshape = np.append(imageMat.ravel(), key)
        csv_writer.writerow(imageMat_reshape.tolist())
