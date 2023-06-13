import tensorflow as tf
import numpy as np
import csv
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

filename = "D:\Bangkit_2023\Capstone_Project\Journey_Project\journey_project\dataset\Dataset_Disabilitas_500.csv"

attributes = ["disability_type", "skills_one", "skills_two", "position"]  

data = []

with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        item = {}
        for attribute in attributes:
            item[attribute] = row[attribute]
        data.append(item)

# Split the data into training set and test set
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  

# Print the training set
print("Training set:")
for item in train_data:
    print(item)

# Print the test set
print("Test set:")
for item in test_data:
    print(item)

# Extract the text features from the data
train_text = np.array([data['disability_type'] + ' ' + data['skills_one'] + ' ' + data['skills_two'] + ' ' + data['position'] for data in train_data])
test_text = np.array([data['disability_type'] + ' ' + data['skills_one'] + ' ' + data['skills_two'] + ' ' + data['position'] for data in test_data])

# Convert the labels to numerical values
train_labels = [data['position'] for data in train_data]
test_labels = [data['position'] for data in test_data]

# Convert the labels to one-hot encoded vectors
unique_labels = list(set(train_labels + test_labels))
num_labels = len(unique_labels)

label_mapping = {label: i for i, label in enumerate(unique_labels)}
train_labels_encoded = [label_mapping[label] for label in train_labels]
test_labels_encoded = [label_mapping[label] for label in test_labels]

train_labels_encoded = np.array(train_labels_encoded, dtype=np.int32)
test_labels_encoded = np.array(test_labels_encoded, dtype=np.int32)

train_labels_one_hot = tf.keras.utils.to_categorical(train_labels_encoded, num_labels)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels_encoded, num_labels)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)

train_sequences = tokenizer.texts_to_sequences(train_text)
test_sequences = tokenizer.texts_to_sequences(test_text)

# Pad sequences to ensure equal length
max_seq_length = 100  
train_sequences = pad_sequences(train_sequences, maxlen=max_seq_length)
test_sequences = pad_sequences(test_sequences, maxlen=max_seq_length)

# Hyperparameter tuning
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=len(tokenizer.word_index) + 1, 
                               output_dim=hp.Int('embedding_dim', min_value=32, max_value=512, step=32),
                               input_length=max_seq_length))
    model.add(layers.LSTM(hp.Int('lstm_units', min_value=32, max_value=512, step=32)))
    model.add(layers.Dense(num_labels, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # number of different hyperparameter combinations to try
    executions_per_trial=3,  # number of times to train each model, to average out the validation accuracy
    directory='D:\Bangkit_2023\Capstone_Project\Journey_Project\journey_project',  # directory to store the logs and trained models
    project_name='journey_hyperparameter_tuning')  

# Run the hyperparameter search
tuner.search(train_sequences, train_labels_one_hot,
             epochs=10,
             validation_data=(test_sequences, test_labels_one_hot))

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Optimal Embedding Dimension: {best_hps.get('embedding_dim')}")
print(f"Optimal LSTM Units: {best_hps.get('lstm_units')}")
print(f"Optimal Learning Rate: {best_hps.get('learning_rate')}")

# Define the early stopping criteria
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    min_delta=0.001,  # Minimum change to consider as improvement
    patience=5,  # Number of epochs with no improvement to stop training
    restore_best_weights=True,  # Restore the best weights recorded during training
)

# Define the model architecture
embedding_dim = 320  # Define the dimensionality of the word embeddings

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_seq_length))
model.add(LSTM(224))
model.add(Dense(num_labels, activation='softmax'))

# Compile the model
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 20

model.fit(train_sequences, train_labels_one_hot, batch_size=batch_size, epochs=epochs, validation_data=(test_sequences, test_labels_one_hot), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(test_sequences, test_labels_one_hot)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Recommender System
new_data = {
    'disability_type': 'Hearing Impairment',
    'skills_one': 'Adobe Illustrator',
    'skills_two': 'UI UX Design',
    'position': 'Unknown'
}

new_text = [new_data['disability_type'] + ' ' + new_data['skills_one'] + ' ' + new_data['skills_two'] + ' ' + new_data['position']]

new_sequences = tokenizer.texts_to_sequences(new_text)
new_sequences = pad_sequences(new_sequences, maxlen=max_seq_length)

predictions = model.predict(new_sequences)
predicted_label_indices = np.argsort(predictions[0])[::-1][:10]
ranked_labels = [unique_labels[index] for index in predicted_label_indices]

# Sequential output according to rank
output = "[" + ", ".join(ranked_labels) + "]"
print(output)
