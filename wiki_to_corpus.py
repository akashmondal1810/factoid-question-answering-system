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
    cat_file = open("data/categories.txt", 'r')
    final_cat_file = open("data/wiki_categories.txt", 'w')

    sp = sentence_parser()

    all_cats = []
    cat_file.readline()
    
    for line in cat_file:
        cat = line.strip()
        all_cats.append(cat)

    cat_file.close()

    track = {'s':0, 'f':0}
    missing = []
    num_sent = 0

    for cat in all_cats:
        print(cat)
        sents = []

        try:
            sents = sent_tokenize(wikipedia.summary(cat))

        except:
            try:
               sents = sent_tokenize(wikipedia.summary(cat, auto_suggest=False))

            except:
                track['f'] += 1
                missing.append(cat)
                continue
                
        for s in sents:
            if len(s) < 5:
                continue
            
            text = sp.wiki_sub(s.strip(),cat)
            text = sp.parse_sentence(text)
            corpus_file.write(' '.join(text) + '\n')
            num_sent += 1
            
        track['s'] += 1
        final_cat_file.write(cat + "\n")

##        except:
##            print("Unexpected error:", sys.exc_info()[0])
##            raise
        

    print("Success: " + str(track['s']))
    print("Fail: " + str(track['f']))
    print("Missing pages:" + ' ,'.join(missing))
    print("Number of sentences: " + str(num_sent))

##    print(all_cats)

    corpus_file.close()
    final_cat_file.close()
