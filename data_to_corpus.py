# Processes sentences from training data:
#   - removes stopwords
#   - stems words

# Also creates a file listing all words in the vocabulary and their
# frequency in the corpus. (To be changed.)


from csv import DictReader, DictWriter
from collections import defaultdict

from numpy import array
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

kID = 'id'
kQUESTION = 'question'
kANSWER = 'correctAnswer'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'




if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("data/sci_train.csv", 'r')))
    test = list(DictReader(open("data/sci_test.csv", 'r')))

    corpus_file = open("data/corpus.txt", 'w')
    vocab_file = open("data/vocab.txt", 'w')

##    print(len(train))
##    print(len(test))

    stemmer = SnowballStemmer("english")
    stopWords = set(stopwords.words('english'))

    vocab_count = defaultdict(lambda: defaultdict(int))
    categories = defaultdict(int)
    vocab = set()


    for x in train:
        #sent = x[kQUESTION].lower().split()
        sent = word_tokenize(x[kQUESTION].lower())

        wordsFiltered = []

         # remove stopwords
        for w in sent:
            if w not in stopWords:
                wordsFiltered.append(w)

        # stem
        sent = [stemmer.stem(word) for word in wordsFiltered]

        # write process sentence
        corpus_file.write(' '.join(sent) + '\n')

        correct = 'answer' + x[kANSWER]
        categories[x[correct]] += 1

        for w in sent:
            vocab.add(w)
            vocab_count[w]['total'] += 1
            vocab_count[w][correct] += 1

    vocab = list(vocab)
    vocab.sort()

    # write vocab to file
    for w in vocab:
        vocab_file.write(w + ' ' + str(vocab_count[w]['total']) + '\n')

    vocab_file.close()
    corpus_file.close()
