import numpy as np
from logisticregression import LogisticRegression


class word2vec:
    def __init__(self, data, embedding):
        self.data = data
        self.window_size = 5
        self.embedding_size = 1000
        self.embedding = embedding
        self.unique = {}
        self.neg_table = None

    def find_positive_context(self, index):
        positive_context = {}
        for i in range(max(0, index - self.window_size), min(index + self.window_size + 1, len(self.data))):
            if i != index:
                positive_context[self.data[i]] = self.embedding[self.data[i]]
        return positive_context

    def find_negative_context(self, index, positive_context):
        # negative sampling
        counter = 0
        negative_context = {}
        k = self.unique[self.data[index]] //  self.window_size + 1
        while counter < self.window_size * k:
            neg = np.random.choice(self.neg_table)
            if self.data[neg] not in positive_context.keys() and neg != index:
                negative_context[self.data[neg]] = self.embedding[self.data[neg]]
                counter += 1
        return negative_context

    def count_unique_word(self):
        for word in self.data:
            if word in self.unique:
                self.unique[word] += 1
            else:
                self.unique[word] = 1
        vocab = list(self.unique.keys())
        frequencies = np.array([self.unique[word] for word in vocab], dtype=np.float32)
        probabilities = frequencies ** 0.75
        probabilities /= probabilities.sum()
        self.neg_table = np.random.choice(len(vocab), size=1000000, p=probabilities)

    def make_random_embedding(self):
        for word in set(self.data):
            if word not in self.embedding.keys():
                self.embedding[word] = np.random.randn(self.embedding_size)


    def learn_embedding(self):
        self.count_unique_word()
        self.make_random_embedding()
        for i in range(len(self.data)):
            if i % 1000 == 0:
                print(i)
            pos = self.find_positive_context(i)
            neg = self.find_negative_context(i, pos)
            new_pos, new_neg, new_w = LogisticRegression().logistic_regression(pos, neg, self.embedding[self.data[i]])
            self.embedding[self.data[i]] = new_w
            for word in new_pos.keys():
                self.embedding[word] = new_pos[word]
            for word in new_neg.keys():
                self.embedding[word] = new_neg[word]
