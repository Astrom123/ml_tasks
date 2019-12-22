from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from tensorflow.python.keras.backend import set_session
from keras import layers
import numpy as np
import tensorflow as tf
import pickle
import json

graph = tf.get_default_graph()
sess = tf.Session()
set_session(sess)


class HumorClassifier:

    def __init__(self):

        with open("params.json", "r") as f:
            params = json.load(f)
            self.max_len = int(params['max_len'])

        self.model = create_model(params)
        self.model.load_weights('weights.h5')
        self.model._make_predict_function()

        with open('tokenizer.bin', 'br') as f:
            self.tokenizer = pickle.load(f)

    def is_humorous(self, text):
        global graph
        global sess

        text = np.array([text])
        text = self.tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, padding='post', maxlen=self.max_len)
        with graph.as_default():
            set_session(sess)
            return self.model.predict_classes(text)[0][0]


def create_model(params):
    embedding_dim = 256
    max_len = int(params['max_len'])
    vocab_size = int(params['vocab_size'])

    filter_sizes = [2, 3, 5, 7]
    conv_filters = []

    input_tensor = layers.Input(shape=(max_len,))
    input_layer = layers.Embedding(vocab_size, embedding_dim, input_length=70)(input_tensor)

    for f_size in filter_sizes:
        conv_filter = layers.Conv1D(128, f_size, activation='relu')(input_layer)
        conv_filter = layers.GlobalMaxPooling1D()(conv_filter)
        conv_filters.append(conv_filter)

    conc_layer = layers.Concatenate()(conv_filters)
    graph = Model(inputs=input_tensor, outputs=conc_layer)

    model = Sequential()
    model.add(graph)
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(conv_filters), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
