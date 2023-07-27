import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

df = pd.read_csv('the_enchanted_april.csv')

processed_articles = df["document_processed"].tolist()

processed_articles = [item for item in processed_articles if isinstance(item, str)]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_articles)

# Create input sequences and labels for training
input_sequences = []
for sentence in processed_articles:
    sequence = tokenizer.texts_to_sequences([processed_articles])[0]
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to have the same length
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Split input sequences into input X and target y
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# Convert target y to one-hot encoded vectors
total_words = len(tokenizer.word_index)
y = np.eye(total_words)[y]

# Build the LSTM model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_length -1))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='loss', mode= 'min', verbose=1, patience=0)

mc = ModelCheckpoint('/Users/Thulana/Desktop/AIMT/Academics/AIMT_2023_S_term_3/Software_Tools_Emerging_Technologies/Project/pLLM_LSTM_Git',monitor='accuracy', mode= 'max', verbose=1, save_best_only=True)

# Train the model
model.fit(X, y, epochs=10, verbose=1, callbacks=(es, mc))

model.save('the_enchanted_april.h5')