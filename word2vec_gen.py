#code for converting word to vector
#uses gensim library's word2vec implementations
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk import tokenize

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
 
if __name__ == '__main__':
    print("word2vec done")
    data_path = "data\\sci_sample.csv"

    sentences = wordvec(data_path)
    model = Word2Vec(sentences)
    print(model.most_similar('A', topn=5))
    # saving the model - optional
    # model.save("data/model/word2vec_gensim")
    # model.wv.save_word2vec_format("data/model/word2vec_org",
                                  # "data/model/vocabulary",
                                  # binary=False)