---
# 🚀 Capstone-Bangkit-2023-Journey

## 💼 About Journey, Recommender System and Content Based Filtering
Journey (Job and Empowerment for Disability) uses a TensorFlow model for content-based filtering to recommend job listings to users. 👩‍💼 The model learns from job attributes like disability type, positions, and skills to create embeddings representing the similarity between jobs. 📚 

Content-based filtering is a type of recommendation system that works with data that the user provides, either explicitly (rating) or implicitly (clicking on a link). It uses the data to build a model of the user's preferences, then recommends items that are similar to those that the user has shown a preference for in the past. 📊

This repository contains code for hyperparameter tuning of a deep learning model using TensorFlow and Keras. The model is trained to predict positions based on various attributes, such as disability type, skills, and position. The hyperparameter tuning is performed using the Kerastuner library. 🧪

## 🗂 Dataset
The data used to train and test our model is contained in a CSV file named `Dataset_Disabilitas_500.csv` 📝 The file includes four different attributes: disability_type, skills_one, skills_two, and position. We've split the dataset into a training and a testing set following an 80:20 ratio to ensure unbiased model evaluation. 🏋️‍♀️

## 🧰 Requirement
To run the code in this repository, the following dependencies are required:

1. TensorFlow
2. NumPy
3. scikit-learn
4. Keras Tuner

## 🏗 Model Architecture
The model architecture is defined in the build_model function. It consists of an embedding layer, LSTM layer, and a dense output layer with softmax activation. The hyperparameters for the model, such as embedding dimension, LSTM units, and learning rate, are tuned using random search with the Kerastuner library. 🧬

## 🔍 Hyperparameter Tuning
The hyperparameter tuning is performed using the RandomSearch tuner. It searches for the best hyperparameters by evaluating multiple trials of the model with different hyperparameter configurations. The tuner searches for the optimal values of embedding dimension, LSTM units, and learning rate. The best hyperparameters are then used to build the final model. 🎯

## 🏃‍♂️ Training
The model is trained using the optimal hyperparameters obtained from the tuning process. The training is performed on the training set for a specified number of epochs. The model is compiled with the Adam optimizer and categorical cross-entropy loss. Early stopping is applied to prevent overfitting. ⏱️

## 📝 Evaluation
After training, the model is evaluated on the test set to measure its performance. The test loss and accuracy are reported.
``` 
Test Loss: 1.0059 
Test Accuracy: 0.8929 

```

## 💾 Saving the Model and Deployment
The provided code combines Flask and TensorFlow Serving to create a web API for serving predictions using a trained TensorFlow model. Flask handles the incoming POST requests, validates the data, and preprocesses it. TensorFlow Serving hosts the model and performs predictions. The code loads the tokenizer, model, and unique labels during initialization. The `/predict` endpoint preprocesses the input, passes it to the model for prediction, and returns the ranked labels as a JSON response. To deploy the code using TensorFlow Serving, the model needs to be exported in the SavedModel format, and TensorFlow Serving should be installed and configured to serve the model. 🚀
