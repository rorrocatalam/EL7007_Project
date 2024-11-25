# -----------------------------------------------------------------------------
#  Imports
# -----------------------------------------------------------------------------
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNet
from tensorflow.keras import layers


# -----------------------------------------------------------------------------
# Modelos
# -----------------------------------------------------------------------------

base_model_vgg16 = VGG16(weights='imagenet')
model_vgg16      = Model(inputs=base_model_vgg16.input,
                         outputs=base_model_vgg16.get_layer('fc2').output)

base_model_resnet50 = ResNet50(weights='imagenet')
model_resnet50      = Model(inputs=base_model_resnet50.input,
                            outputs=base_model_resnet50.get_layer('avg_pool').output)

base_model_inception = InceptionV3(weights='imagenet')
model_inception      = Model(inputs=base_model_inception.input,
                             outputs=base_model_inception.get_layer('avg_pool').output)


base_model_mobilenet = MobileNet(weights='imagenet')
model_mobilenet      = Model(inputs=base_model_mobilenet.input,
                             outputs=base_model_mobilenet.get_layer('global_average_pooling2d').output)


# -----------------------------------------------------------------------------
# Extraccion de caracteristicas
# -----------------------------------------------------------------------------

def get_features(model_name, db_path, eye, res_path):
    global model_vgg16
    global model_resnet50
    global model_inception
    global model_mobilenet

    # Seleccion del modelo
    if model_name == 'vgg16':
        model = model_vgg16
        target_size = (224, 224)
    elif model_name == 'resnet50':
        model = model_resnet50
        target_size = (299, 299)
    elif model_name == 'inception':
        model = model_inception
        target_size = (224, 224)
    elif model_name == 'mobilenet':
        model = model_mobilenet
        target_size = (224, 224)
    else:
        print('Modelos disponibles: [vgg16, resnet50, inception, mobilenet]')
        return
    
    # Tamannos y nombres de las salidas
    layer_outputs = []
    layer_names   = []

    # Seleccion de salidas
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D, layers.Dense)):
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)

    # Inicializacion de archivos csv con caracteristicas
    create_csv_files(layer_names, res_path)

    # Modelo con salidas en todas las capas
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    i=0
    # Se recorre el directorio con usuarios
    for user_dir in os.listdir(db_path):
        user_path = os.path.join(db_path, user_dir)
        
        # Directorio del ojo seleccionado
        target_path = os.path.join(user_path, eye)
        if os.path.exists(target_path):
            
            # Directorio de cada imagen
            for image_file in os.listdir(target_path):
                image_path = os.path.join(target_path, image_file)
                user = get_user(image_path)
                
                # Extraccion y almacenamiento de caracteristicas
                image_ft   = get_image_features(image_path, activation_model, target_size)
                write_csv_files(layer_names, res_path, image_ft, user)
                i+=1
                print(i)
        
    print(f'Features guardadas en \"{res_path}\"')

def get_image_features(img_path, model, target_size):
    # Lectura de imagen como tensor
    img        = image.load_img(img_path, color_mode='grayscale', target_size=target_size)
    img_array  = image.img_to_array(img)
    img_rgb    = np.repeat(img_array, 3, axis=2)
    img_tensor = np.expand_dims(img_rgb, axis=0)

    # Prediccion con el modelo
    activations = model.predict(img_tensor)
    # Promedio canales para reducir features
    act_mean    = [np.mean(activations[i], axis=3, keepdims=False) if len(activations[i].shape) == 4 
                   else activations[i] for i in range(len(activations))]
    # Aplanamiento para guardar en csv
    act_flatten = [np.ravel(arr) for arr in act_mean]
    return act_flatten

def create_csv_files(name_list, res_path):
    # Creacion del directorio si no existe
    os.makedirs(res_path, exist_ok=True)

    # Creacion de archivos csv
    for name in name_list:
        file_path = os.path.join(res_path, f"{name}.csv")
        with open(file_path, 'w') as file:
            pass

def write_csv_files(name_list, res_path, ft_list, user):
    # Recorrido por archivos con features
    for name, features in zip(name_list, ft_list):
        file_path =  os.path.join(res_path, f"{name}.csv")

        # Escribir cada linea en cada archivo
        with open(file_path, 'a') as file:
            line = ','.join(map(str, features))
            line = ','.join(map(str, [user] + list(features)))
            file.write(line + '\n')
        
def get_user(img_path):
    parts = os.path.normpath(img_path).split(os.sep)
    index = parts.index("CASIA-IrisV3")
    return parts[index+2]


model_name = 'vgg16'

db_path = '../CASIA-IrisV3/CASIA-Iris-Lamp'
sel_eye = 'L'
res_path = f'../features_{sel_eye}/{model_name}'

print('Saving VGG16 features...')
get_features(model_name, db_path, sel_eye, res_path)