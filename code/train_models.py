import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def train_svm_per_layer(ft_path, output_file, out5_file, n_features=8193):
    # Archivos csv con caracteristicas
    csv_files  = [file for file in os.listdir(ft_path) if file.endswith('.csv')]
    for i in range(len(csv_files)):
        # Archivo
        csv_file = f'{ft_path}/{csv_files[i]}'
    
        # Cantidad de caracteristicas
        with open(csv_file, 'r') as archivo:
            reader = csv.reader(archivo)
            rows   = next(reader)
            n_cols = len(rows)
            print(f'Reading \"{csv_file}\" with {n_cols} features')
        
        if n_cols < n_features:
            # Sujetos y caracteristicas
            print('Loading features...')
            subjects = np.loadtxt(csv_file, delimiter=',', usecols=0, converters={0: lambda s: int(float(s))})
            features = np.loadtxt(csv_file, delimiter=',', usecols=range(1, np.loadtxt(csv_file, delimiter=',').shape[1]))

            # Separacion de datos
            print('Creating datasets...')
            X_train, X_test, y_train, y_test = train_test_split(features, subjects, test_size=0.3, random_state=42, stratify=subjects)
            # Normalizacion ajustada con datos de entrenamiento
            print('Normalizing data...')
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s  = scaler.transform(X_test)

            # Entrenamiento SVM y prediccion
            print('Training SVM...')
            svm = SVC(C=100, gamma='auto', kernel='sigmoid', probability=True)
            svm.fit(X_train_s, y_train)
            # Prediccion y guardo resultados
            print('Saving results...')
            probabilities = svm.predict_proba(X_test_s)
            rank1_acc = top_k_accuracy_score(y_test,probabilities, k=1)
            rank5_acc= top_k_accuracy_score(y_test,probabilities, k=5)
            with open(output_file, 'a') as f:
                layer_name = os.path.splitext(os.path.basename(csv_file))[0]
                f.write(f"{layer_name},{rank1_acc:.4f}\n")
                print(f'Layer {layer_name}: {100*rank1_acc:.2f}% Accuracy')
            with open(out5_file, 'a') as f:
                layer_name = os.path.splitext(os.path.basename(csv_file))[0]
                f.write(f"{layer_name},{rank5_acc:.4f}\n")
                print(f'Layer {layer_name}: {100*rank5_acc:.2f}% Accuracy Rank5')
            print()

def train_rf_per_layer(ft_path, output_file, out5_file, n_features=8193):
    # Archivos csv con caracteristicas
    csv_files  = [file for file in os.listdir(ft_path) if file.endswith('.csv')]
    for i in range(len(csv_files)):
        # Archivo
        csv_file = f'{ft_path}/{csv_files[i]}'

        # Cantidad de caracteristicas
        with open(csv_file, 'r') as archivo:
            reader = csv.reader(archivo)
            rows   = next(reader)
            n_cols = len(rows)
            print(f'Reading \"{csv_file}\" with {n_cols} features')
        
        if n_cols < n_features:
            # Sujetos y caracteristicas
            print('Loading features...')
            subjects = np.loadtxt(csv_file, delimiter=',', usecols=0, converters={0: lambda s: int(float(s))})
            features = np.loadtxt(csv_file, delimiter=',', usecols=range(1, np.loadtxt(csv_file, delimiter=',').shape[1]))

            # Separacion de datos
            print('Creating datasets...')
            X_train, X_test, y_train, y_test = train_test_split(features, subjects, test_size=0.3, random_state=42, stratify=subjects)
            # Normalizacion ajustada con datos de entrenamiento
            print('Normalizing data...')
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s  = scaler.transform(X_test)

            # Entrenamiento SVM y prediccion
            print('Training RF...')
            rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=1, max_features=9)
            # svm = SVC(C=100, gamma='auto', kernel='sigmoid')
            rf.fit(X_train_s, y_train)
            # Prediccion y guardo resultados
            print('Saving results...')
            probs = rf.predict_proba(X_test_s)
            rank1_acc = top_k_accuracy_score(y_test,probs, k=1)
            rank5_acc= top_k_accuracy_score(y_test,probs, k=5)
            with open(output_file, 'a') as f:
                layer_name = os.path.splitext(os.path.basename(csv_file))[0]
                f.write(f"{layer_name},{rank1_acc:.4f}\n")
                print(f'Layer {layer_name}: {100*rank1_acc:.2f}% Accuracy')
            with open(out5_file, 'a') as f:
                layer_name = os.path.splitext(os.path.basename(csv_file))[0]
                f.write(f"{layer_name},{rank5_acc:.4f}\n")
                print(f'Layer {layer_name}: {100*rank5_acc:.2f}% Accuracy Rank5')
            print()