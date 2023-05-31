import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Read dataset from CSV
df = pd.read_csv('datasets/asde.csv')

# Extract relevant columns for clustering
features = df[['skills_one', 'skills_two', 'is_hearing_impairment', 'is_visual_impairment',
               'is_physical_disability', 'is_all_types_of_disabilities', 'is_neurodiversity',
               'is_mobility_impairment', 'is_speech_impairment', 'is_cognitive_disability',
               'is_learning_disability']]

# Convert categorical columns to numeric using one-hot encoding
features = pd.get_dummies(features)

# Standardize the dataset
scaler = StandardScaler()
dataset = scaler.fit_transform(features)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
dataset = pca.fit_transform(dataset)

# TensorFlow clustering
input_dim = dataset.shape[1]
latent_dim = 2  # Number of latent dimensions

# Define the encoder
encoder_input = Input(shape=(input_dim,))
encoder_output = Dense(latent_dim)(encoder_input)
encoder_model = Model(encoder_input, encoder_output)

# Define the decoder
decoder_input = Input(shape=(latent_dim,))
decoder_output = Dense(input_dim)(decoder_input)
decoder_model = Model(decoder_input, decoder_output)

# Define the autoencoder
autoencoder_input = Input(shape=(input_dim,))
autoencoder_output = decoder_model(encoder_model(autoencoder_input))
autoencoder_model = Model(autoencoder_input, autoencoder_output)

# Compile the autoencoder
autoencoder_model.compile(optimizer=Adam(), loss='mse')

# Train the autoencoder
autoencoder_model.fit(dataset, dataset, epochs=100, batch_size=2)

# Perform clustering
latent_vectors = encoder_model.predict(dataset)
k = 2  # Number of clusters
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=k)
kmeans.train(lambda: tf.compat.v1.estimator.inputs.numpy_input_fn(x=latent_vectors, shuffle=True)(), steps=100)

# Get cluster assignments
cluster_assignments = list(kmeans.predict_cluster_index(lambda: tf.compat.v1.estimator.inputs.numpy_input_fn(x=latent_vectors, shuffle=False)()))

# Add cluster assignments to the dataframe
df['cluster'] = cluster_assignments

# Print cluster assignments
print(df[['skills_one', 'skills_two', 'position', 'cluster']])
