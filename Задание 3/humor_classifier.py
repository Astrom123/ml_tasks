from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import pickle


class HumorClassifier:

    def __init__(self):
        self.model = load_model('resources//model.h5')
        self.maxlen = self.model.layers[0].input_shape[1]

        with open('resources//tokenizer.bin', 'br') as f:
            self.tokenizer = pickle.load(f)

    def is_humorous(self, text):
        text = np.array([text])
        text = self.tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, padding='post', maxlen=self.maxlen)
        return self.model.predict_classes(text)[0][0]
