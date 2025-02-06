import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Define sequence parameters
max_len = 75  # Maximum sequence length
vocab_size = 100  # Example vocabulary size
num_labels = 5  # Number of output labels

# Build the model
def build_model():
    input_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")  # Ensure correct dtype
    embedding = Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True, name="embedding")(input_layer)  

    # Add an explicit masking layer (alternative to mask_zero=True in Embedding)
    masked_input = Masking(mask_value=0.0)(embedding)  

    lstm = LSTM(64, return_sequences=True, name="lstm")(masked_input)  

    # Do NOT explicitly pass mask to TimeDistributed
    output_layer = TimeDistributed(Dense(num_labels, activation="softmax"), name="time_distributed")(lstm)  

    model = Model(input_layer, output_layer)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Build and summarize model
model = build_model()
model.summary()

# Dummy training data
X_train = np.random.randint(1, vocab_size, (100, max_len))  # 100 samples with random word indices
y_train = np.random.randint(0, num_labels, (100, max_len, 1))  # Random labels for each word
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_labels)  # One-hot encode labels

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
