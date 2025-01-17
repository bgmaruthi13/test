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


###########################################################################################################


### 1. What are Large Language Models (LLMs)? List any four limitations/drawbacks of LLMs.  
**Answer**:  
LLMs are AI models trained on extensive text data to perform various language tasks, including translation, summarization, and answering questions.
They use deep learning architectures like transformers to process and generate human-like language. Examples include GPT-4, BERT, and LLaMA.  

**Limitations**:  
1. **Resource Intensive**: Training and deploying LLMs require massive computational resources, leading to high costs and environmental concerns.  
2. **Bias in Outputs**: They can reproduce and amplify biases present in their training data.  
3. **Lack of Contextual Understanding**: They may generate plausible but incorrect or nonsensical answers due to a lack of true understanding.  
4. **Inability to Perform Complex Reasoning**: LLMs struggle with tasks requiring deep logical reasoning or multi-step problem-solving.  

---

### 2. Discuss RNN cell and its drawback. How does LSTM overcome RNN drawbacks?  
**Answer**:  
RNN (Recurrent Neural Network) cells process sequential data by maintaining a hidden state across time steps. 
They work well for tasks like language modeling and time series prediction. 
However, RNNs suffer from the **vanishing gradient problem**, where gradients diminish during backpropagation, making it difficult to learn long-term dependencies. 
This issue limits their effectiveness on lengthy sequences.  

**LSTM** (Long Short-Term Memory) networks overcome this drawback by introducing gated mechanisms:  
1. **Forget Gate**: Controls which information to remove from the memory.  
2. **Input Gate**: Determines what new information to add.  
3. **Output Gate**: Regulates the output based on the updated memory.  
These gates enable LSTMs to retain important information over long sequences, improving their performance on tasks like speech recognition and machine translation.  

---

### 3. Explain Named Entity Recognition (NER) with an example.  
**Answer**:  
NER identifies and classifies named entities (e.g., names of people, places, organizations, dates) within text into predefined categories. 
It is a crucial task in Natural Language Processing (NLP) for extracting structured information from unstructured text.  

**Example**:  
Input: *“Amazon was founded by Jeff Bezos in 1994.”*  
NER Output:  
- **Amazon**: Organization  
- **Jeff Bezos**: Person  
- **1994**: Date  

Applications include question-answering systems, chatbots, and content summarization. NER helps streamline information retrieval by extracting key details from large datasets.  

---

### 4. What is Generative AI? Difference between discriminative and generative AI.  
**Answer**:  
Generative AI refers to models that create new data similar to their training data. These models generate text, images, videos, or music, with examples like GPT, DALL-E, and Stable Diffusion.  

**Difference**:  
- **Discriminative AI**: Learns decision boundaries between classes and is used for classification tasks (e.g., Logistic Regression, SVMs).  
- **Generative AI**: Models the underlying data distribution to generate samples and is used for tasks like text generation, image creation, and synthetic data generation (e.g., GANs, Variational Autoencoders).  

---

### 5. Explain the drawbacks of LSTM.  
**Answer**:  
LSTM (Long Short-Term Memory) networks are widely used for sequential data, but they have limitations:  
1. **High Computational Cost**: Their complex gating mechanisms increase training and inference times.  
2. **Memory Constraints**: Handling very long sequences can still overwhelm the memory cells.  
3. **Overfitting**: LSTMs require careful tuning of hyperparameters to avoid overfitting, especially on small datasets.  
4. **Parallelization Limitations**: Unlike transformers, LSTMs process sequences step-by-step, making them slower for large-scale tasks.  

---

### 6. Draw the transformer architecture and explain the attention mechanism.  
**Answer**:  
Transformers use self-attention mechanisms to process sequences in parallel, unlike RNNs and LSTMs.  
1. **Attention Mechanism**:  
   - Each input word is represented as a query (Q), key (K), and value (V).  
   - Attention score = Softmax(QKᵀ / √d) * V.  
   - This enables the model to focus on relevant parts of the sequence.  

2. **Architecture**:  
   - Consists of an encoder-decoder stack.  
   - The encoder processes input data, and the decoder generates output based on attention weights and encoded representations.  

**Example**: In the sentence “She threw the ball,” the attention mechanism links "threw" strongly to "ball," understanding their relationship.  

---

### 7. What is zero-shot learning?  
**Answer**:  
Zero-shot learning allows AI models to perform tasks they haven't been explicitly trained on by leveraging generalized knowledge from pretraining.
It uses natural language descriptions or semantic representations to connect unseen tasks to existing knowledge.  

**Example**: A zero-shot model trained on text classification can infer sentiment for a dataset of tweets, even without being directly trained on social media data. 
This enables efficient task generalization.  

---

### 8. What is Generative AI? List two concerns and approaches to mitigate them.  
**Answer**:  
Generative AI creates realistic content, such as text, images, and audio, based on learned patterns from training data.  

**Concerns**:  
1. **Misinformation**: Can produce fake or misleading content, contributing to misinformation campaigns.  
2. **Bias Amplification**: May perpetuate societal biases present in training data.  

**Mitigation Approaches**:  
1. **Robust Training**: Use diverse and high-quality datasets with fairness-aware techniques.  
2. **Content Verification**: Implement watermarks, cryptographic signatures, or AI-generated content detectors to verify authenticity.  

---

### 9. What is attention in Transformer architecture? Example to compute attention scores.  
**Answer**:  
Attention in transformers helps the model focus on relevant words in a sequence, improving context understanding.  

**Example Calculation**:  
Sentence: “I love AI.”  
1. Compute Query (Q), Key (K), and Value (V) matrices for each word.  
2. Compute Attention scores = Softmax(QKᵀ / √d).  
3. Multiply scores with Value (V) for the weighted output.  

The word “love” will have a strong focus on “AI,” ensuring the model understands the sentiment and context effectively.  

---

### 10. What is Prompt Engineering? List two prompting approaches.  
**Answer**:  
Prompt Engineering involves designing input prompts to guide AI models toward desired outcomes. 
It plays a crucial role in maximizing the effectiveness of LLMs in various tasks.  

**Approaches**:  
1. **Few-shot Prompting**: Include a few examples of the desired task within the prompt to help the model generalize.  
2. **Chain-of-Thought Prompting**: Encourage the model to generate step-by-step reasoning for complex tasks.  

Effective prompting significantly improves the accuracy and relevance of AI-generated outputs.

