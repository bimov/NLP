import numpy as np
from collections import defaultdict
from logisticregression import LogisticRegression


def get_ngrams(text, n):
    """Формирует список всех n-грамм в тексте"""
    ngrams = []
    for word in text.split():
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i + n])
    return ngrams


def train_tokenizer(text, num_merges=10000, max_ngram=8):
    """Обучает токенизатор, заменяя частые n-граммы токенами"""
    vocab = {word: 1 for word in text.split()}  # Начальные слова
    token_map = {}  # Карта замен токенов

    for _ in range(num_merges):
        # Собираем статистику частот n-грамм
        ngram_counts = defaultdict(int)
        for n in range(2, max_ngram + 1):  # Берем n-граммы разной длины
            for word in vocab:
                for ngram in get_ngrams(word, n):
                    ngram_counts[ngram] += vocab[word]

        if not ngram_counts:
            break

        # Выбираем самую частую n-грамму
        best_ngram = max(ngram_counts, key=ngram_counts.get)
        token = f"[{best_ngram}]"  # Новый токен (можно заменить на спецсимвол)

        # Обновляем словарь, заменяя n-грамму новым токеном
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = word.replace(best_ngram, token)
            new_vocab[new_word] = freq

        vocab = new_vocab
        token_map[token] = best_ngram  # Запоминаем соответствие токена n-грамме
        if _ % 1 == 0:
            print(_)

    return vocab, token_map


def tokenize(text, token_map):
    """Токенизирует текст, используя обученные n-граммы"""
    tokens = []
    words = text.split()

    for word in words:
        for token, ngram in sorted(token_map.items(), key=lambda x: -len(x[1])):
            word = word.replace(ngram, token)
        tokens.append(word)

    return tokens


class Word2Vec:
    def __init__(self, data, embedding=None, embedding_size=1000, window_size=5, num_merges=10000, max_ngram=8):
        self.raw_data = data
        self.window_size = window_size
        self.embedding_size = embedding_size
        if embedding is None:
            self.embedding = {}
        else:
            self.embedding = embedding
        self.unique = {}
        self.neg_table = None

        # 🔹 Обучаем n-граммный токенизатор
        self.vocab, self.token_map = train_tokenizer(" ".join(data), num_merges, max_ngram)
        self.data = self.tokenize_data(data)

    def tokenize_data(self, data):
        """🔹 Токенизирует весь корпус данных"""
        return [tokenize(sentence, self.token_map) for sentence in data]

    def find_positive_context(self, index):
        """Находит положительный контекст (окно вокруг слова)"""
        positive_context = {}
        for i in range(max(0, index - self.window_size), min(index + self.window_size + 1, len(self.data))):
            if i != index:
                positive_context[self.data[i]] = self.embedding[self.data[i]]
        return positive_context

    def find_negative_context(self, index, positive_context):
        """🔹 Отрицательный сэмплинг"""
        counter = 0
        negative_context = {}
        k = self.unique[self.data[index]] // self.window_size
        while counter < self.window_size * k:
            neg = np.random.choice(self.neg_table)
            if self.data[neg] not in positive_context.keys() and neg != index:
                negative_context[self.data[neg]] = self.embedding[self.data[neg]]
                counter += 1
        return negative_context

    def count_unique_word(self):
        """🔹 Подсчет частоты токенов"""
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
        """🔹 Создает случайные вектора эмбеддингов"""
        for word in set(sum(self.data, [])):  # Теперь данные в токенизированном формате
            if word not in self.embedding:
                self.embedding[word] = np.random.randn(self.embedding_size)

    def learn_embedding(self):
        """🔹 Обучение word2vec"""
        self.count_unique_word()
        self.make_random_embedding()

        for i in range(len(self.data)):
            if i % 1000 == 0:
                print(f"Обработано: {i} / {len(self.data)}")

            pos = self.find_positive_context(i)
            neg = self.find_negative_context(i, pos)
            new_pos, new_neg, new_w = LogisticRegression().logistic_regression(pos, neg, self.embedding[self.data[i]])

            self.embedding[self.data[i]] = new_w
            for word in new_pos:
                self.embedding[word] = new_pos[word]
            for word in new_neg:
                self.embedding[word] = new_neg[word]
