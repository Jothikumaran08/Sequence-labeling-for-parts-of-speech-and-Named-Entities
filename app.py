import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset

# Load the dataset (this loads the conll2003 dataset for NER)
dataset = load_dataset("conll2003")
train_data = dataset["train"]

# Preprocess the dataset to get train_sentences and train_labels
train_sentences = [[word for word in sentence["tokens"]] for sentence in train_data]
train_labels = [[label for label in sentence["ner_tags"]] for sentence in train_data]

# Build word2idx and label2idx dictionaries
word2idx = {word: i + 2 for i, word in enumerate(set(word for sentence in train_sentences for word in sentence))}
word2idx["PAD"] = 0
word2idx["UNK"] = 1
idx2word = {i: word for word, i in word2idx.items()}

# Create label mappings
unique_labels = set(label for sublist in train_labels for label in sublist)
label2idx = {label: i for i, label in enumerate(unique_labels)}
idx2label = {i: label for label, i in label2idx.items()}

# Make sure to check the labels being used
st.write(f"Label Mappings: {idx2label}")

max_len = 75  # Maximum sequence length for padding

# Load the trained model
model = load_model("ner_model.h5")  # Ensure you saved your model as 'ner_model.h5'

# Function to predict entities
def predict_entities(text):
    # Tokenize the input sentence - this should match the preprocessing during training
    sentence = text.split()  # Basic space-based tokenization (or use another tokenizer if trained with one)
    
    # Check tokenization
    st.write(f"Tokenized input: {sentence}")
    
    # Preprocess the sentence (convert words to indices and pad)
    X = pad_sequences([[word2idx.get(w, 1) for w in sentence]], padding="post", maxlen=max_len)
    
    # Predict named entities
    preds = model.predict(X)
    
    # Get the index of the highest prediction for each word
    preds = np.argmax(preds, axis=-1)[0]
    
    # Debugging: Print raw predictions
    st.write("Raw Predictions (Indices):", preds)
    
    # Map predictions back to labels and format output
    predictions = [(word, idx2label.get(pred, 'O')) for word, pred in zip(sentence, preds)]
    formatted_predictions = [f"{word}: {label}" for word, label in predictions]
    
    return formatted_predictions

# Streamlit interface
st.title("Named Entity Recognition (NER) App")
st.write("Enter a sentence to get the named entities detected:")

# Input area for text
user_input = st.text_area("Enter your text:")

# When the user enters text, predict and show the entities
if user_input:
    entities = predict_entities(user_input)
    st.write("Predicted Named Entities:")
    for entity in entities:
        st.write(f"**{entity}**")
