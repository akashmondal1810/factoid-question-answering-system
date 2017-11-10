# Processes sentences from training data:
#   - removes stopwords
#   - stems words

# Also creates a file listing all words in the vocabulary and their
# frequency in the corpus. (To be changed.)


from csv import DictReader, DictWriter
from collections import defaultdict

from data_to_corpus import sentence_parser

from numpy import array
import numpy as np

import string

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import sys
import wikipedia

kID = 'id'
kQUESTION = 'question'
kANSWER = 'correctAnswer'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'



if __name__ == "__main__":

    # Cast to list to keep it all in memory
    corpus_file = open("data/wiki_corpus.txt", 'w')
    cat_file = open("data/categories.csv", 'r')

    sp = sentence_parser()

    all_cats = []
    for line in cat_file:
        words = line.split(',')
        all_cats.append(words[0])

    cat_file.close()

    track = {'s':0, 'f':0}
    missing = []
    num_sent = 0

    for cat in all_cats:
        print(cat)

        try:
            sents = sent_tokenize(wikipedia.summary(cat))

            for s in sents:
                if len(s) < 5:
                    continue
                
                text = sp.wiki_sub(s.strip(),cat)
                text = sp.parse_sentence(text)
                corpus_file.write(' '.join(text) + '\n')
                num_sent += 1
                
            track['s'] += 1

##        except:
##            print("Unexpected error:", sys.exc_info()[0])
##            raise
        except:
            track['f'] += 1
            missing.append(cat)

    print("Success: " + str(track['s']))
    print("Fail: " + str(track['f']))
    print("Missing pages:" + ' ,'.join(missing))
    print("Number of sentences: " + str(num_sent))

##    print(all_cats)

    corpus_file.close()
