
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


class wordvec(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(self.dirname):
            sline = line.strip()
            if sline == "":
                continue
            tokenized_line = word_tokenize(sline)
            yield tokenized_line


class w2v():
    def __init__(self, data_path="data/corpus.txt"):  # "data/corpus.txt"):
        # Train the model
        self.size = 100
        sentences = wordvec(data_path)

        self.model = Word2Vec(sentences, size=self.size,
                              window=5, min_count=5, workers=4)
        self.vocab = list(self.model.wv.vocab.keys())
        self.null = np.float64(np.zeros(self.size))

        self.stemmer = SnowballStemmer("english")
        self.stopwords = set(stopwords.words('english'))

    def wv(self, word):

        # remove stopwords
        if word in self.stopwords:
            return self.null

        # stem
        word = self.stemmer.stem(word)

        if word in self.vocab:
            return np.float64(self.model[word])
        else:
            return self.null

    def sv(self, sline):
        word_list = sline.split()
        N = len(word_list)
        v = self.null
        if N == 0:
            return v
        for word in word_list:
            v = v + np.multiply(self.wv(word), (1.0 / N))
        return v


if __name__ == '__main__':
    model = w2v()
    print(model.sv('what is real ?'))
