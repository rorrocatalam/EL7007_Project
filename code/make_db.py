import os
import cv2
import numpy as np
from get_iris import get_iris

# Original dataset folder (not in repo)
folder_or = 'CASIA1'
# Folder with the new dataset
folder_db = 'database'

# Resolution of the iris images
radpixels   = 20
angulardiv  = 240

# Recorrer la carpeta de origen
for root, dirs, files in os.walk(folder_or):
    for file in files:
        if file.endswith('.jpg'):
            # Image path
            img_path = os.path.join(root, file)
            # Relative path of the image
            img_relpath = os.path.relpath(img_path)
            # New image with iris
            img_iris = get_iris(img_relpath, radpixels=radpixels,
                                angulardiv=angulardiv)
            img_iris = (img_iris*255).astype(np.uint8)
            # Subrelative path to origin folder
            subimg_relpath = os.path.relpath(root, folder_or)
            
            # Destination folder to save the new image
            fol_despath = os.path.join(folder_db, subimg_relpath)
            os.makedirs(fol_despath, exist_ok=True)
            
            # Path of the new image
            img_despath = os.path.join(fol_despath, file)
            cv2.imwrite(img_despath, img_iris)
            print(f'Image saved in: {img_despath}')
