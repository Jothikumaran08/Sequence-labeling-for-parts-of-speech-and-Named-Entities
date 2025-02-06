import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset

# Load the model
model = load_model("ner_model.h5")

# Load the dataset for preprocessing
dataset = load_dataset("conll2003")
train_data = dataset["train"]

# Preprocess the dataset to build word2idx and label2idx dictionaries
train_sentences = [[word for word in sentence["tokens"]] for sentence in train_data]
word2idx = {word: i + 2 for i, word in enumerate(set(word for sentence in train_sentences for word in sentence))}
word2idx["PAD"] = 0
word2idx["UNK"] = 1

unique_labels = set(label for sublist in train_data["ner_tags"] for label in sublist)
label2idx = {label: i for i, label in enumerate(unique_labels)}
idx2label = {i: label for label, i in label2idx.items()}

max_len = 75  # Maximum sequence length for padding

# Function to predict entities
def predict_entities(text):
    sentence = text.split()
    X = pad_sequences([[word2idx.get(w, 1) for w in sentence]], padding="post", maxlen=max_len)
    preds = model.predict(X)
    preds = np.argmax(preds, axis=-1)[0]
    
    # Map predictions back to labels and format output
    predictions = [(word, idx2label.get(pred, 'O')) for word, pred in zip(sentence, preds)]
    
    # Format output to match user expectations
    formatted_output = {word: label for word, label in predictions}
    return formatted_output

# Streamlit interface
st.title("Named Entity Recognition (NER) App")
st.write("Enter a sentence to get the named entities detected:")

user_input = st.text_area("Enter your text:")

if user_input:
    entities = predict_entities(user_input)
    st.write("Predicted Named Entities:")
    for word, label in entities.items():
        st.write(f"**{word}: {label}**")
