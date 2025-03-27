from data import get_words, cleaned_word
from word2vec import word2vec
from word2vec_with_tokenizer import Word2Vec
import numpy as np
from NN import NN
import math


def compute_perplexity(model, data, window_size, embedding, unique):
    N = len(data) - window_size
    log_prob_sum = 0.0

    for i in range(N):
        context = data[i: i + window_size]
        print(context)
        true_word = data[i + window_size]
        pred = model.predict(context)[1]
        true_index = unique[true_word]
        prob = pred[0][true_index]
        if prob <= 0:
            prob = 1e-10
        log_prob_sum += math.log(prob)
    perplexity = math.exp(-log_prob_sum / N)
    return perplexity


def learn_embedding(text=get_words()):
    a = word2vec(text, np.load('embeddings10.npy', allow_pickle=True).item())
    a.learn_embedding()
    np.save('embeddings10.npy', a.embedding, allow_pickle=True)


def learn_embedding_with_token(text=get_words()):
    a = Word2Vec(text)
    a.learn_embedding()
    np.save('embeddings10_token.npy', a.embedding, allow_pickle=True)

def load_data():
    embedding = np.load('embeddings10.npy', allow_pickle=True).item()
    unique = embedding.keys()
    uniques = {}
    count = 0
    for word in unique:
        uniques[word] = count
        count += 1
    return embedding, uniques


def predict():
    text = input('Enter your sentence: ').split()
    for i in range(len(text)):
        text[i] = cleaned_word(text[i])
    text = [word for word in text if word != '']
    embedding, unique = load_data()
    for word in text:
        if word not in unique:
            learn_embedding(text)
            embedding, unique = load_data()
            break

    neural_network = NN(embedding, unique, get_words(), 5, True)
    predict_word = neural_network.predict(text[-5:])[0]
    print("Предсказанное слово: ", predict_word)
    print(compute_perplexity(neural_network, text[-5:] + [predict_word], 5, embedding, unique))


def train():
    embedding, unique = load_data()
    neural_network = NN(embedding, unique, get_words(), 5, True)
    print(neural_network.train())
    neural_network.model.save('neural_network3.h5')

if __name__ == '__main__':
    predict()