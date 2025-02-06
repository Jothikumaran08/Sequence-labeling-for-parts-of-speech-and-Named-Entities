import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("conll2003")
train_data = dataset["train"]

# Preprocess the dataset
train_sentences = [[word for word in sentence["tokens"]] for sentence in train_data]
train_labels = [[label for label in sentence["ner_tags"]] for sentence in train_data]

# Build word2idx and label2idx dictionaries
word2idx = {word: i + 2 for i, word in enumerate(set(word for sentence in train_sentences for word in sentence))}
word2idx["PAD"] = 0
word2idx["UNK"] = 1
idx2word = {i: word for word, i in word2idx.items()}

unique_labels = set(label for sublist in train_labels for label in sublist)
label2idx = {label: i for i, label in enumerate(unique_labels)}
idx2label = {i: label for label, i in label2idx.items()}

# Prepare data for training
max_len = 75
X = pad_sequences([[word2idx.get(w, 1) for w in sentence] for sentence in train_sentences], padding="post", maxlen=max_len)
y = pad_sequences([[label2idx[label] for label in labels] for labels in train_labels], padding="post", maxlen=max_len)

# Build the model
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=len(word2idx), output_dim=128, mask_zero=True)(input_layer)
masked_input = Masking(mask_value=0.0)(embedding)
lstm = Bidirectional(LSTM(64, return_sequences=True))(masked_input)
output_layer = TimeDistributed(Dense(len(label2idx), activation="softmax"))(lstm)

model = Model(input_layer, output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X, np.expand_dims(y, -1), epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save("ner_model.h5")
