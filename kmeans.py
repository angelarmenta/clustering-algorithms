#developed by Roberto Ángel Meléndez-Armenta
#https://www.youtube.com/@educar-ia

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar dataset
file_path = 'RUTA DEL DATASET'
data_limones = pd.read_csv(file_path)

# Visualizar dataset
plt.scatter(data_limones['peso'], data_limones['diametro'])
plt.xlabel('peso')
plt.ylabel('diametro')
plt.title('Dataset Limones y Naranjas')
plt.show()

# Número of clusters
kmeans = KMeans(n_clusters=2)

# Ajustar el modelo de KMeans
kmeans.fit(data_limones[['peso', 'diametro']])

# Etiquetar los datos
labels = kmeans.labels_

# Obtener los centroides
centers = kmeans.cluster_centers_

# Visualizar los clusters
plt.scatter(data_limones['peso'], data_limones['diametro'], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red')
plt.xlabel('peso')
plt.ylabel('diametro')
plt.title('KMeans (Dataset Limones y Naranjas)')
plt.show()

# Predecir nuevos valores
nuevo_dato = np.array([[150, 7]])
nuevo_grupo = kmeans.predict(nuevo_dato)
print('El nuevo dato pertenece al cluster:', nuevo_grupo[0])

# Agregar las etiquetas al dataset
data_limones['cluster'] = labels

# Guardar el nuevo dataset (incluye etiquetas) en un nuevo archivo CSV
data_limones.to_csv('RUTA DEL NUEVO ARCHIVO', index=False)
