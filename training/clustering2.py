import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Read dataset from CSV for job seekers
job_seeker_df = pd.read_csv('datasets/asde.csv')

# Extract relevant columns for clustering
job_seeker_features = job_seeker_df[['skills_one', 'skills_two', 'is_hearing_impairment', 'is_visual_impairment',
                                     'is_physical_disability', 'is_all_types_of_disabilities', 'is_neurodiversity',
                                     'is_mobility_impairment', 'is_speech_impairment', 'is_cognitive_disability',
                                     'is_learning_disability']]

# Convert categorical columns to numeric using one-hot encoding
job_seeker_features = pd.get_dummies(job_seeker_features)

# Standardize the job seeker dataset
scaler = StandardScaler()
job_seeker_dataset = scaler.fit_transform(job_seeker_features)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
job_seeker_dataset = pca.fit_transform(job_seeker_dataset)

# TensorFlow clustering for job seekers
input_dim = job_seeker_dataset.shape[1]
latent_dim = 2  # Number of latent dimensions

# Define the encoder for job seekers
job_seeker_encoder_input = Input(shape=(input_dim,))
job_seeker_encoder_output = Dense(latent_dim)(job_seeker_encoder_input)
job_seeker_encoder_model = Model(job_seeker_encoder_input, job_seeker_encoder_output)

# Define the decoder for job seekers
job_seeker_decoder_input = Input(shape=(latent_dim,))
job_seeker_decoder_output = Dense(input_dim)(job_seeker_decoder_input)
job_seeker_decoder_model = Model(job_seeker_decoder_input, job_seeker_decoder_output)

# Define the autoencoder for job seekers
job_seeker_autoencoder_input = Input(shape=(input_dim,))
job_seeker_autoencoder_output = job_seeker_decoder_model(job_seeker_encoder_model(job_seeker_autoencoder_input))
job_seeker_autoencoder_model = Model(job_seeker_autoencoder_input, job_seeker_autoencoder_output)

# Compile the autoencoder for job seekers
job_seeker_autoencoder_model.compile(optimizer=Adam(), loss='mse')

# Train the autoencoder for job seekers
job_seeker_autoencoder_model.fit(job_seeker_dataset, job_seeker_dataset, epochs=100, batch_size=2)

# Perform clustering for job seekers
job_seeker_latent_vectors = job_seeker_encoder_model.predict(job_seeker_dataset)
job_seeker_k = 2  # Number of clusters for job seekers
job_seeker_kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=job_seeker_k)
job_seeker_kmeans.train(lambda: tf.compat.v1.estimator.inputs.numpy_input_fn(x=job_seeker_latent_vectors, shuffle=True)(), steps=100)

# Get cluster assignments for job seekers
job_seeker_cluster_assignments = list(job_seeker_kmeans.predict_cluster_index(lambda: tf.compat.v1.estimator.inputs.numpy_input_fn(x=job_seeker_latent_vectors, shuffle=False)()))

# Add cluster assignments to the job seeker dataframe
job_seeker_df['cluster'] = job_seeker_cluster_assignments


# Read dataset from CSV for user
user_df = pd.read_csv('datasets/user.csv')

# Extract relevant columns for clustering
user_features = user_df[['skills_one', 'skills_two', 'disability_type']]

# Convert categorical columns to numeric using one-hot encoding
user_features = pd.get_dummies(user_features)

# Standardize the user dataset
user_dataset = scaler.transform(user_features)

# Reduce dimensionality using PCA
user_dataset = pca.transform(user_dataset)

# Perform clustering for user
user_latent_vectors = job_seeker_encoder_model.predict(user_dataset)
user_cluster_assignments = list(job_seeker_kmeans.predict_cluster_index(lambda: tf.compat.v1.estimator.inputs.numpy_input_fn(x=user_latent_vectors, shuffle=False)()))

# Add cluster assignments to the user dataframe
user_df['cluster'] = user_cluster_assignments


# Print job seeker cluster assignments
print("Job Seeker Clusters:")
print(job_seeker_df[['skills_one', 'skills_two', 'cluster']])

# Print user cluster assignments
print("\nUser Clusters:")
print(user_df[['skills_one', 'skills_two', 'disability_type', 'cluster']])
