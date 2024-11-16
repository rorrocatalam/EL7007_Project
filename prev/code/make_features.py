import os
import numpy as np
from feature_extractor import *

db_foldername = 'database'
ft_foldername = 'features'

# CSV files to write
vgg16_csv     = os.path.join(ft_foldername, 'vgg16.csv')
resnet_csv    = os.path.join(ft_foldername, 'resnet.csv')
inception_csv = os.path.join(ft_foldername, 'inception.csv')
mobilenet_csv = os.path.join(ft_foldername, 'mobilenet.csv')

def save_features(csv_file, subject, features):
    features_with_subject = np.insert(features, 0, subject)
    with open(csv_file, 'a') as f:
        np.savetxt(f, [features_with_subject], delimiter=',', fmt='%.6f')
        
for root, dirs, files in os.walk(db_foldername):
    for file in files:
        if file.endswith('.jpg'):
            # Image path
            img_path = os.path.join(root, file)
            # Relative path of the image
            img_relpath = os.path.relpath(img_path)
            
            # Subject
            subject = int(os.path.split(os.path.dirname(img_relpath))[-1])
            
            # Feature extractor
            f_vgg16     = feature_extractor_VGG16(img_relpath)
            f_resnet    = feature_extractor_ResNet50(img_path)
            f_inception = feature_extractor_InceptionV3(img_relpath)
            f_mobilenet = feature_extractor_MobileNet(img_relpath)
            
            # Save features to CSV files
            save_features(vgg16_csv, subject, f_vgg16)
            save_features(resnet_csv, subject, f_resnet)
            save_features(inception_csv, subject, f_inception)
            save_features(mobilenet_csv, subject, f_mobilenet)
            