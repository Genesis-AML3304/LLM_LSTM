import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('the_enchanted_april.h5')

df = pd.read_csv('the_enchanted_april.csv')
processed_articles = df["document_processed"].tolist()
processed_articles = [item for item in processed_articles if isinstance(item, str)]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_articles)

def gentext(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word

    print(seed_text)

seed_text = 'Not for her were mediaeval castles'
next_words = 10
gentext(seed_text, next_words)