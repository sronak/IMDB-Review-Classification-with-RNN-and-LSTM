# IMDB Movie Review Sentiment Analysis with LSTM
This is a Python script for performing sentiment analysis on IMDB movie reviews using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) with TensorFlow.

The dataset used is the IMDB movie review dataset which contains 50,000 movie reviews (25,000 for training and 25,000 for testing), labeled as positive or negative. The goal of this project is to develop a model that can accurately classify the sentiment of a given review as either positive or negative.

## Results
The model achieves an accuracy of '83.09' on the test data, which is a good performance for this task. A classification report is generated to show the precision, recall, f1-score, and support for each class.

## Dependencies
This project requires the following dependencies:

- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Keras
- Scikit-learn

## Instructions for Running the Code
- Clone the repository using ```git clone https://github.com/sronak/IMDB-Review-Classification-with-RNN-and-LSTM.git```
- Install the required libraries using ```pip install -r requirements.txt```
- Run the Python script ```IMDB Review Classification with RNN and LSTM.ipynb```

## Future Improvements
Here are some possible improvements that could be made to this project:

- Fine-tune the hyperparameters of the model to achieve better performance.
- Experiment with different RNN architectures, such as bidirectional LSTM or GRU.
- Use pre-trained word embeddings, such as GloVe or Word2Vec, to improve the performance of the model.
- Use a larger dataset to train the model and evaluate its performance.
- Build a web application that allows users to enter movie reviews and get a sentiment prediction from the trained model.
