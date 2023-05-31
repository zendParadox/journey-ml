import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Membaca dataset dari file CSV
dataset = pd.read_csv('datasets/asde.csv')

# Memisahkan atribut target (misalnya, 'position') dari atribut yang digunakan untuk pengelompokan
target = dataset['position']
features = dataset.drop('position', axis=1)

# Preprocessing dataset
# ...

# Konversi dataset menjadi numpy array
dataset_array = features.values

# Standarisasi dataset
scaler = StandardScaler()
dataset_array = scaler.fit_transform(dataset_array)

# Reduksi dimensi menggunakan PCA
pca = PCA(n_components=2)
dataset_array = pca.fit_transform(dataset_array)

# TensorFlow Clustering
input_dim = dataset_array.shape[1]
latent_dim = 2  # Jumlah dimensi laten

# Definisikan encoder
encoder_input = Input(shape=(input_dim,))
encoder_output = Dense(latent_dim)(encoder_input)
encoder_model = Model(encoder_input, encoder_output)

# Definisikan decoder
decoder_input = Input(shape=(latent_dim,))
decoder_output = Dense(input_dim)(decoder_input)
decoder_model = Model(decoder_input, decoder_output)

# Definisikan autoencoder
autoencoder_input = Input(shape=(input_dim,))
autoencoder_output = decoder_model(encoder_model(autoencoder_input))
autoencoder_model = Model(autoencoder_input, autoencoder_output)

# Kompilasi autoencoder
autoencoder_model.compile(optimizer=Adam(), loss='mse')

# Latih autoencoder
autoencoder_model.fit(dataset_array, dataset_array, epochs=100, batch_size=2)

# Melakukan pengelompokan
latent_vectors = encoder_model.predict(dataset_array)
k = 2  # Jumlah kelompok
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=k)
kmeans.train(lambda: tf.compat.v1.estimator.inputs.numpy_input_fn(x=latent_vectors, shuffle=True)(), steps=100)

# Dapatkan penugasan kelompok
cluster_assignments = list(kmeans.predict_cluster_index(lambda: tf.compat.v1.estimator.inputs.numpy_input_fn(x=latent_vectors, shuffle=False)()))

# Cetak penugasan kelompok
for i, cluster in enumerate(cluster_assignments):
    print("Data point:", dataset_array[i], "Kelompok:", cluster)
