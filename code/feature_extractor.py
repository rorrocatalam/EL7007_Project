# -----------------------------------------------------------------------------
#  Imports
# -----------------------------------------------------------------------------
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNet
from tensorflow.keras.applications.vgg16 import preprocess_input as ppi_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as ppi_resnet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as ppi_inception
from tensorflow.keras.applications.mobilenet import preprocess_input as ppi_mobilenet

# -----------------------------------------------------------------------------
# VGG16
# -----------------------------------------------------------------------------

base_model_vgg16 = VGG16(weights='imagenet')
model_vgg16      = Model(inputs=base_model_vgg16.input,
                         outputs=base_model_vgg16.get_layer('fc2').output)

def feature_extractor_VGG16(path):
    '''
    Obtiene caracteristicas de una imagen en blanco y negro con VGG16.
    Retorna un vector con 4096 elementos.
    '''
    img = image.load_img(path, color_mode='grayscale', target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_rgb = np.repeat(img_array, 3, axis=2)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = ppi_vgg16(img_rgb)
    return model_vgg16.predict(img_rgb)[0]

# -----------------------------------------------------------------------------
# ResNet50
# -----------------------------------------------------------------------------

base_model_resnet50 = ResNet50(weights='imagenet')
model_resnet50      = Model(inputs=base_model_resnet50.input,
                            outputs=base_model_resnet50.get_layer('avg_pool').output)

def feature_extractor_ResNet50(path):
    '''
    Obtiene caracteristicas de una imagen en blanco y negro con ResNet50.
    Retorna un vector con 2048 elementos.
    '''
    img = image.load_img(path, color_mode='grayscale', target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_rgb = np.repeat(img_array, 3, axis=2)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = ppi_resnet50(img_rgb)
    return model_resnet50.predict(img_rgb)[0]

# -----------------------------------------------------------------------------
# InceptionV3
# -----------------------------------------------------------------------------

base_model_inception = InceptionV3(weights='imagenet')
model_inception      = Model(inputs=base_model_inception.input,
                             outputs=base_model_inception.get_layer('avg_pool').output)

def feature_extractor_InceptionV3(path):
    '''
    Obtiene características de una imagen en blanco y negro con InceptionV3.
    Retorna un vector con 2048 elementos.
    '''
    img = image.load_img(path, color_mode='grayscale', target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_rgb = np.repeat(img_array, 3, axis=2)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = ppi_inception(img_rgb)
    return model_inception.predict(img_rgb)[0]

# -----------------------------------------------------------------------------
# MobileNet
# -----------------------------------------------------------------------------

base_model_mobilenet = MobileNet(weights='imagenet')
model_mobilenet      = Model(inputs=base_model_mobilenet.input,
                             outputs=base_model_mobilenet.get_layer('global_average_pooling2d').output)

def feature_extractor_MobileNet(path):
    '''
    Obtiene características de una imagen en blanco y negro con MobileNet.
    Retorna un vector con 1024 elementos.
    '''
    img = image.load_img(path, color_mode='grayscale', target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_rgb = np.repeat(img_array, 3, axis=2)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = ppi_mobilenet(img_rgb)
    return model_mobilenet.predict(img_rgb)[0][0,0,:]