# -----------------------------------------------------------------------------
#  Imports
# -----------------------------------------------------------------------------
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNet
from tensorflow.keras.applications.vgg16 import preprocess_input as ppi_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as ppi_resnet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as ppi_inception
from tensorflow.keras.applications.mobilenet import preprocess_input as ppi_mobilenet
from tensorflow.keras import layers



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




# -----------------------------------------------------------------------------
# Layer Reader
# -----------------------------------------------------------------------------
def print_layers(n_model, img_tensor):
    global model_inception
    global model_mobilenet
    global model_resnet50
    global model_vgg16
    names=models  = ['inception', 'mobilenet', 'resnet50','vgg16']
    models  = [model_inception, model_mobilenet, model_resnet50, model_vgg16]
    model = models[n_model]
    layer_outputs = []
    layer_names = []
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)   
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros(((size + 1) * n_cols - 1,
                                images_per_row * (size + 1) - 1))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_index = col * images_per_row + row
                channel_image = layer_activation[0, :, :, channel_index].copy()
                if channel_image.sum() != 0:
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[
                    col * (size + 1): (col + 1) * size + col,
                    row * (size + 1) : (row + 1) * size + row] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.axis("off")
        plt.imshow(display_grid, aspect="auto", cmap="viridis")
        os.makedirs(f'HiddenLayers/{names[n_model]}', exist_ok=True)
        plt.savefig(f'HiddenLayers/{names[n_model]}/{layer_name}')
        plt.close()

img_path = 'database/1/001_1_1.jpg'
img = image.load_img(img_path, color_mode='grayscale', target_size=(224, 224))
img_array = image.img_to_array(img)
img_rgb = np.repeat(img_array, 3, axis=2)
img_rgb = np.expand_dims(img_rgb, axis=0)
print_layers(1, img_rgb)
print_layers(2, img_rgb)
print_layers(3, img_rgb)

img = image.load_img(img_path, color_mode='grayscale', target_size=(299, 299))
img_array = image.img_to_array(img)
img_rgb = np.repeat(img_array, 3, axis=2)
img_rgb = np.expand_dims(img_rgb, axis=0)
print_layers(0, img_rgb)