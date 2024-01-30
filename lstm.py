import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


from preprocess import preprocessed_text


def classify_from_file(file_path):
    df = pd.read_excel(file_path)

    df['Processed_Text'] = df['Teks'].apply(preprocessed_text)

    model = load_model('lstm_model (1).h5')
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    X_seq = tokenizer.texts_to_sequences(df['Processed_Text'])
    X_pad = pad_sequences(X_seq, maxlen=100, padding="pre")

    predictions = model.predict(X_pad)
    predicted_labels = np.argmax(predictions, axis=1)

    class_labels = [-1, 0, 1]
    predicted_category_labels = [class_labels[i] for i in predicted_labels]

    text_class_labels = ['Negatif', 'Netral', 'Positif']
    predicted_text_labels = [text_class_labels[i] for i in predicted_labels]

    df['Category_Label'] = predicted_category_labels
    df['Predicted_Label'] = predicted_text_labels
    return df


def classify_text(input_text):
    processed_text = preprocessed_text(input_text)

    model = load_model('lstm_model6.h5')
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequences, maxlen=100, padding='pre')
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction)

    class_labels = ['Negatif', 'Netral', 'Positif']
    predicted_category_label = class_labels[predicted_label]

    return predicted_category_label
