#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from gensim.models import Word2Vec

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')

# Initializing the PorterStemmer
ps = PorterStemmer()

# Importing the dataset
dataset = pd.read_csv('data.csv')  # Replace with your dataset file name
text_column = 'text'  # Feature column
target_column = 'airline_sentiment'  # Target column

# Cleaning the text column
corpus = []
for i in range(0, len(dataset)):
    review = dataset[text_column][i].lower()
    review = re.sub(r'http\S+|www\S+', '', review)  # Remove URLs
    review = re.sub('[^a-zA-Z]', ' ', review)  # Remove punctuation
    review = review.split()
    all_stopwords = set(stopwords.words('english'))
    review = [ps.stem(word) for word in review if word not in all_stopwords]  # Remove stopwords and apply stemming
    review = ' '.join(review)
    corpus.append(review)

# Encoding the target variable
le = LabelEncoder()
y = le.fit_transform(dataset[target_column].values)

# --- Logistic Regression --- #
# Converting the text column into numerical data using Count-Vectorization
count_vectorizer = CountVectorizer(max_features=1500)
X_count = count_vectorizer.fit_transform(corpus).toarray()

# Converting the text column into numerical data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
X_tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()

# Get the vocabulary of the TF-IDF vectorizer
vocabulary = tfidf_vectorizer.vocabulary_

# Print the vocabulary (first 20 terms for example)
print("Vocabulary (first 20 terms):")
for term, index in list(vocabulary.items())[:20]:
    print(f"{term}: {index}")

# Split both datasets into training and testing sets (stratified sampling)
X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_count, X_tfidf, y, test_size=0.25, random_state=0, stratify=y
)

# Logistic Regression on Count-Vectorized data
log_reg_count = LogisticRegression(max_iter=1000)
log_reg_count.fit(X_train_count, y_train)
y_pred_count = log_reg_count.predict(X_test_count)
accuracy_count = accuracy_score(y_test, y_pred_count)
print("Logistic Regression Accuracy (Count-Vectorized):", accuracy_count)

# Logistic Regression on TF-IDF data
log_reg_tfidf = LogisticRegression(max_iter=1000)
log_reg_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = log_reg_tfidf.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print("Logistic Regression Accuracy (TF-IDF):", accuracy_tfidf)

# Comparison of both models
if accuracy_count > accuracy_tfidf:
    print("Count-Vectorization model has better accuracy.")
else:
    print("TF-IDF model has better accuracy.")

# --- Naive Bayes --- #
# Train a Naive Bayes classifier on the training set (using Count Vectorized data)
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_count, y_train)
y_pred_nb = naive_bayes_model.predict(X_test_count)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Model Accuracy: {accuracy_nb}")

# Print the classification report for Naive Bayes model
print("Naive Bayes Model Classification Report:")
print(classification_report(y_test, y_pred_nb))


# --- Word2Vec (Skip-Gram and CBOW) --- #
# Preprocessing for Word2Vec
corpus2 = []
for i in range(0, len(dataset)):
    review = dataset['text'][i].lower()
    review = re.sub(r'http\S+|www\S+', '', review)  # Remove URLs
    review = re.sub('[^a-zA-Z]', ' ', review)  # Remove punctuation
    review = review.split()
    all_stopwords = set(nltk.corpus.stopwords.words('english'))
    review = [word for word in review if word not in all_stopwords]  # Remove stopwords
    corpus2.append(review)
    

# Step 1: Training the Skip-Gram Word2Vec model
skipgram_model = Word2Vec(sentences=corpus2, vector_size=100, window=5, sg=1, min_count=1)

# Fetching the top 5 most similar words for the word 'food' using the Skip-Gram model
similar_words_skipgram = skipgram_model.wv.most_similar('food', topn=5)
print("\nTop 5 Similar Words (Skip-Gram) for 'food':")
for word, similarity in similar_words_skipgram:
    print(f"{word}: {similarity}")

# Step 2: Training the CBOW Word2Vec model
cbow_model = Word2Vec(sentences=corpus2, vector_size=100, window=5, sg=0, min_count=1)

# Fetching the top 5 most similar words for the word 'food' using the CBOW model
similar_words_cbow = cbow_model.wv.most_similar('food', topn=5)
print("\nTop 5 Similar Words (CBOW) for 'food':")
for word, similarity in similar_words_cbow:
    print(f"{word}: {similarity}")

# Step 3: Compare the outputs of Skip-Gram and CBOW
if similar_words_skipgram == similar_words_cbow:
    print("\nBoth models give the same results.")
else:
    print("\nThe models give different results.")


# In[ ]:


# Importing the necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, RNN

# Download necessary NLTK datasets
nltk.download('stopwords')

# Initializing the PorterStemmer
ps = PorterStemmer()

# Importing the dataset
dataset = pd.read_csv('data.csv')  # Replace with your dataset file name
text_column = 'text'  # Feature column
target_column = 'airline_sentiment'  # Target column

# Cleaning the text column
corpus = []
for i in range(0, len(dataset)):
    review = dataset[text_column][i].lower()
    review = re.sub(r'http\S+|www\S+', '', review)  # Remove URLs
    review = re.sub('[^a-zA-Z]', ' ', review)  # Remove punctuation
    review = review.split()
    all_stopwords = set(stopwords.words('english'))
    review = [ps.stem(word) for word in review if word not in all_stopwords]  # Remove stopwords and apply stemming
    review = ' '.join(review)
    corpus.append(review)

# Encoding the target variable
le = LabelEncoder()
y = le.fit_transform(dataset[target_column].values)

# --- LSTM Model Building --- #
# Step a: Convert the cleaned text into sequences using Tokenizer
tokenizer = Tokenizer(num_words=1500)
tokenizer.fit_on_texts(corpus)
X_sequences = tokenizer.texts_to_sequences(corpus)

# Padding sequences to ensure all inputs have the same length
max_sequence_length = 100  # Choose a max length based on your dataset
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

# Step b: Build and compile an LSTM model for sentiment analysis
model = Sequential()

# Add an embedding layer to convert text data to numeric format
model.add(Embedding(input_dim=1500, output_dim=50, input_length=max_sequence_length))

# Add an LSTM layer with 100 units, a ReLU activation function, and dropout for regularization
model.add(LSTM(100, activation='relu', dropout=0.2))

# Add a fully connected Dense output layer with softmax activation for classification
model.add(Dense(1, activation='sigmoid'))  # For binary classification, change to softmax for multi-class

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step c: Train the model and evaluate its performance on the testing set
# Split data into training and testing sets for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_padded, y, test_size=0.25, random_state=0, stratify=y)

# Train the model
history = model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test_lstm)
print(f'Test Accuracy: {test_accuracy}')

# Predict on the test set
y_pred_lstm = model.predict(X_test_lstm)
y_pred_lstm = (y_pred_lstm > 0.5).astype(int)  # For binary classification (sigmoid activation)

# Print classification report for LSTM model
print("LSTM Model Performance:")
print(classification_report(y_test_lstm, y_pred_lstm))


# # The task specific pretrained transformers pipeline models is saved and provided. Use them to perform below Text processing tasks as questioned below.

# In[ ]:


### Using Sentence Classification - Sentiment Analysis model classification_pipeline_model, classify the sentence 
#   "Such a nice weather outside!" into positive/negative with score.


# In[ ]:


from transformers import pipeline

# Load the pre-trained sentiment analysis pipeline model
classification_pipeline_model = pipeline('sentiment-analysis', model='classification_pipeline_model')

# Input sentence
sentence = "Such a nice weather outside!"

# Perform sentiment classification
result = classification_pipeline_model(sentence)

# Display the result
print(f"Sentence: {sentence}")
print(f"Sentiment: {result[0]['label']}")
print(f"Score: {result[0]['score']:.4f}")


# In[ ]:


## Using Named Entity Recognition model ner_pipeline_model, perform name-entity- recognition of sentence 
# "Hugging Face is a French company based in New-York.


# In[ ]:


from transformers import pipeline

# Load the pre-trained Named Entity Recognition pipeline model
ner_pipeline_model = pipeline('ner', model='ner_pipeline_model', grouped_entities=True)

# Input sentence
sentence = "Hugging Face is a French company based in New-York."

# Perform Named Entity Recognition
ner_results = ner_pipeline_model(sentence)

# Display the results
print(f"Sentence: {sentence}")
print("Named Entities:")
for entity in ner_results:
    print(f"- Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}")


# In[ ]:


## Using the Question Answering model qa_pipeline_model, provide the answer the question asked from the given paragraph 
# (for question and paragraph refer notebook).


# In[ ]:


from transformers import pipeline

# Load the pre-trained Question Answering pipeline model
qa_pipeline_model = pipeline('question-answering', model='qa_pipeline_model')

# Paragraph and question
context = """
Hugging Face is a technology company that focuses on natural language processing (NLP). 
It was founded in 2016 and is known for its open-source library called Transformers, 
which provides tools for machine learning models like BERT, GPT, and others. 
The company has offices in New York and Paris.
"""

question = "Where are the offices of Hugging Face located?"

# Use the QA model to find the answer
qa_result = qa_pipeline_model(question=question, context=context)

# Display the results
print(f"Question: {question}")
print(f"Answer: {qa_result['answer']}")
print(f"Score: {qa_result['score']:.4f}")


# In[ ]:


## Using Text Generation - Mask Filling model tg_pipeline_model, suggest the appropriate words for specified 
#  "MISSING_WORD_Field in the given sentence.


# In[ ]:


from transformers import pipeline

# Load the pre-trained Mask Filling pipeline model
tg_pipeline_model = pipeline('fill-mask', model='tg_pipeline_model')

# Sentence with a missing word represented by the [MASK] token
masked_sentence = "The weather today is absolutely [MASK]."

# Use the Text Generation model to suggest the missing words
suggestions = tg_pipeline_model(masked_sentence)

# Display the suggestions
print(f"Suggestions for the missing word in: '{masked_sentence}'")
for i, suggestion in enumerate(suggestions):
    print(f"{i + 1}. {suggestion['sequence']} (Score: {suggestion['score']:.4f})")


# In[ ]:


## Using Summarization model summarizer pipeline model. provide summarization of the given 
#  Long Tennis Article as provided in notebook.


# In[ ]:


from transformers import pipeline

# Load the pre-trained summarization pipeline model
summarizer_pipeline_model = pipeline("summarization", model="summarizer_pipeline_model")

# The given Long Tennis Article (replace `long_tennis_article` with the actual article text)
long_tennis_article = """
[INSERT LONG TENNIS ARTICLE HERE]
"""

# Perform summarization
summary = summarizer_pipeline_model(long_tennis_article, max_length=150, min_length=40, do_sample=False)

# Display the summarization result
print("Summary of the Long Tennis Article:")
print(summary[0]['summary_text'])


# In[ ]:





# In[ ]:




