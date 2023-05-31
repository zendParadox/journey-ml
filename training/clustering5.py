import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Read dataset from CSV for job seekers
job_seeker_df = pd.read_csv('datasets/dataset_master.csv')

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

# Split the job seeker dataset into training and validation sets
train_size = int(0.8 * len(job_seeker_dataset))
train_data = job_seeker_dataset[:train_size]
val_data = job_seeker_dataset[train_size:]

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

# Train the autoencoder for job seekers with validation
job_seeker_autoencoder_model.fit(train_data, train_data, epochs=100, batch_size=2, validation_data=(val_data, val_data))


# Perform clustering for job seekers
job_seeker_latent_vectors = job_seeker_encoder_model.predict(job_seeker_dataset)
job_seeker_k = 3  # Number of clusters for job seekers
job_seeker_kmeans = KMeans(n_clusters=job_seeker_k, n_init=10)
job_seeker_kmeans.fit(job_seeker_latent_vectors)

# Get cluster assignments for job seekers
job_seeker_cluster_assignments = job_seeker_kmeans.labels_

# Add cluster assignments to the job seeker dataframe
job_seeker_df['cluster'] = job_seeker_cluster_assignments


# Read dataset from CSV for user
user_df = pd.read_csv('datasets/user2.csv')

# Extract relevant columns for clustering
user_features = user_df[['skills_one', 'skills_two', 'disability_type']]

# Convert categorical columns to numeric using one-hot encoding
user_features = pd.get_dummies(user_features)

# Update feature names in user dataset
# Update feature names in user dataset
missing_features = list(set(job_seeker_features.columns) - set(user_features.columns))
new_features = pd.DataFrame(0, index=np.arange(len(user_features)), columns=missing_features)
user_features = pd.concat([user_features, new_features], axis=1)


# Reorder columns to match job seeker dataset
user_features = user_features[job_seeker_features.columns]

# Standardize the user dataset
user_dataset = scaler.transform(user_features)

# Reduce dimensionality using PCA
user_dataset = pca.transform(user_dataset)

# Perform clustering for user
user_latent_vectors = job_seeker_encoder_model.predict(user_dataset)
user_cluster_assignments = job_seeker_kmeans.predict(user_latent_vectors)

# Add cluster assignments to the user dataframe
user_df['cluster'] = user_cluster_assignments


# Recommendation system

# Define function to get recommended job seekers for a given user
def get_recommended_job_seekers(user_id, num_recommendations=5):
    user_row = user_df[user_df['user_id'] == user_id]
    user_cluster = user_row['cluster'].values[0]
    
    recommended_job_seekers = job_seeker_df[job_seeker_df['cluster'] == user_cluster].sample(n=num_recommendations)
    
    return recommended_job_seekers[['skills_one', 'skills_two', 'position']]

# Example usage:
user_id = 100  # Specify the user ID for whom you want recommendations
recommendations = get_recommended_job_seekers(user_id)

# Print the recommended job seekers
print(f"Recommended Job Seekers for User ID {user_id}:")
print(recommendations)
