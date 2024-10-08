# -----------------------------------------------------------------------------
#  Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score,  f1_score, balanced_accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# Lectura de características
# -----------------------------------------------------------------------------

model_name = 'MobileNet' # 'VGG16', 'ResNet50', 'InceptionV3', 'MobileNet'

if model_name == 'VGG16':
    csv_file = 'features/vgg16.csv'
elif model_name == 'ResNet50':
    csv_file = 'features/resnet.csv'
elif model_name == 'InceptionV3':
    csv_file = 'features/inception.csv'
elif model_name == 'MobileNet':
    csv_file = 'features/mobilenet.csv'

# Sujetos
subjects = np.loadtxt(csv_file, delimiter=',', usecols=0, converters={0: lambda s: int(float(s))})
# Caracteristicas
features = np.loadtxt(csv_file, delimiter=',', usecols=range(1, np.loadtxt(csv_file, delimiter=',').shape[1]))
print('Features loaded')

# -----------------------------------------------------------------------------
# Conjuntos de entrenamiento y prueba
# -----------------------------------------------------------------------------

# Separacion de datos
X_train, X_test, y_train, y_test = train_test_split(features, subjects, test_size=0.2, random_state=42, stratify=subjects)
print('Datasets created')
# Normalizacion ajustada con datos de entrenamiento
scaler = StandardScaler()
scaler.fit(X_train)

X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test) 
print('Normalized data\n')

# -----------------------------------------------------------------------------
# Clasificacion con SVM
# -----------------------------------------------------------------------------

print('Finding best parameters for SVM...')
# Modelo SVM
svm = SVC()
# Parametros para GridSearchCV (Regularizacion, kernel y gamma)
params_SVM = {'C': [0.01, 0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
              'gamma': ['scale', 'auto']}
gs_SVM = GridSearchCV(svm, params_SVM, n_jobs=-1)
gs_SVM.fit(X_train_s, y_train)

# Mejores hiperparametros para SVM
C, gamma, kernel = gs_SVM.best_params_['C'], gs_SVM.best_params_['gamma'], gs_SVM.best_params_['kernel']
print(f'Best params: C={C}, kernel={kernel}, gamma={gamma}')

# SVM a entrenar
print('Training SVM...')
svm_b = SVC(C=C, gamma=gamma, kernel=kernel)
svm_b.fit(X_train_s, y_train)

# Prediccion sobre datos de test
y_pred = svm_b.predict(X_test_s)
print(f'Balanced Accuracy = {round(balanced_accuracy_score(y_test, y_pred),4)}')
print(f'Precision = {round(precision_score(y_test, y_pred, average='weighted', zero_division=0),4)}')
print(f'Recall = {round(recall_score(y_test, y_pred, average='weighted', zero_division=0),4)}')
print(f'F1score = {round(f1_score(y_test, y_pred, average='weighted', zero_division=0),4)}')

# Matriz de confusion
conf_mtx_svm = confusion_matrix(y_pred, y_test)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_mtx_svm, annot=False, cmap='Blues', cbar=False)
plt.title(f'Confusion matrix for SVM clasification\nusing {model_name}')
plt.xlabel('Predicted values')
plt.ylabel('Real values')
plt.savefig(f'results/svm_{model_name}.png')
print('Saved results!\n')

# -----------------------------------------------------------------------------
# Clasificacion con Random Forest
# -----------------------------------------------------------------------------

print('Finding best parameters for RF...')
# Modelo Random Forest
rf = RandomForestClassifier()
# Parametros para GridSearchCV
params_RF =  {'n_estimators': [50, 100, 200],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 8],
              'min_samples_leaf': [1, 2, 3, 4],
              'max_features': [5, 7, 9]}
gs_RF = GridSearchCV(rf, params_RF, n_jobs=-1)
gs_RF.fit(X_train_s, y_train)

# Mejores hiperparametros para Random Forest
n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = gs_RF.best_params_['n_estimators'], gs_RF.best_params_['max_depth'], gs_RF.best_params_['min_samples_split'], gs_RF.best_params_['min_samples_leaf'], gs_RF.best_params_['max_features']
print(f'Best params: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, max_features={max_features}')

# RF a entrenar
print('Training RF...\n')
rf_b = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
rf_b.fit(X_train_s, y_train)

# Prediccion sobre datos de test
y_pred = rf_b.predict(X_test_s)
print(f'Balanced Accuracy = {round(balanced_accuracy_score(y_test, y_pred),4)}')
print(f'Precision = {round(precision_score(y_test, y_pred, average='weighted', zero_division=0),4)}')
print(f'Recall = {round(recall_score(y_test, y_pred, average='weighted', zero_division=0),4)}')
print(f'F1score = {round(f1_score(y_test, y_pred, average='weighted', zero_division=0),4)}')

# Matriz de confusion
conf_mtx_svm = confusion_matrix(y_pred, y_test)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_mtx_svm, annot=False, cmap='Blues', cbar=False)
plt.title(f'Confusion matrix for RF clasification\nusing {model_name}')
plt.xlabel('Predicted values')
plt.ylabel('Real values')
plt.savefig(f'results/rf_{model_name}.png')
print('Saved results!\n')
