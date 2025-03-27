import tensorflow.keras as keras
from keras import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model

class NN:
    def __init__(self, embedding, unique, data, window_size, load=False):
        self.embedding = embedding
        self.unique = unique
        self.data = data
        self.embedding_size = len(embedding[self.data[0]])
        self.window_size = window_size
        self.x_train = []
        self.y_train = []
        if load:
            self.model = load_model('neural_network3.h5')
        else:
            self.model = None

    def convert_to_embedding(self, words):
        convert_embedding = []
        for word in words:
            if word not in self.embedding:
                raise ValueError(f"Слово '{word}' отсутствует в словаре эмбеддингов.")
            convert_embedding.append(self.embedding[word])
        return np.array(convert_embedding)

    def data_generator(self, batch_size=4000):
        data_length = len(self.data)
        while True:
            for i in range(0, data_length - self.window_size, batch_size):
                x_batch = []
                y_batch = []
                for j in range(i, min(i + batch_size, data_length - self.window_size)):
                    x = self.convert_to_embedding(self.data[j: j + self.window_size])
                    y_word = self.data[j + self.window_size]
                    one_hot = to_categorical(self.unique[y_word], num_classes=len(self.unique))
                    x_batch.append(x)
                    y_batch.append(one_hot)
                yield np.array(x_batch), np.array(y_batch)

    def model_settings(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(self.window_size, self.embedding_size)))
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dense(len(self.unique.keys()), activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=10, batch_size=32, chunk_size=4000, shuffle=True):
        if self.model is None:
            self.model_settings()

        steps_per_epoch = (len(self.data) - self.window_size) // chunk_size
        if (len(self.data) - self.window_size) % chunk_size != 0:
            steps_per_epoch += 1

        history = self.model.fit(
            self.data_generator(),
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle
        )
        return history

    def predict(self, words):
        if len(words) != self.window_size:
            raise ValueError(f"Длина входного списка должна быть равна window_size ({self.window_size}).")

        for word in words:
            if word not in self.embedding:
                raise ValueError(f"Слово '{word}' отсутствует в словаре эмбеддингов.")

        x = self.convert_to_embedding(words)
        x = x.reshape(1, self.window_size, self.embedding_size)

        pred = self.model.predict(x)
        predicted_index = np.argmax(pred, axis=-1)[0]

        inv_unique = {index: word for word, index in self.unique.items()}
        predicted_word = inv_unique.get(predicted_index, None)

        return predicted_word, pred
