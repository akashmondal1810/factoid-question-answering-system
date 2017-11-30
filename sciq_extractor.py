# 


from csv import DictReader, DictWriter
from collections import defaultdict
import json

from numpy import array
import numpy as np

import string
import random

from data_to_corpus import sentence_parser

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re


kID = 'id'
kQUESTION = 'question'
kANSWER = 'correctAnswer'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'




if __name__ == "__main__":

    new_cats = list(json.load(open('data/answer_set2.json')))
    cat_data = list(DictReader(open("data/categories.txt", 'r')))
    new_q_data = list(json.load(open("data/qb_train.json", 'r')))
    cats = ['_'.join(x['page'].split(' ')) for x in cat_data]
    #cats = cat_data.keys()

    new_cats.sort()
    cats.sort()
    #print(new_cats)
    #print(cats)

    new_cats = set(new_cats)
    cats = set(cats)
    overlap = cats & new_cats

    cats = list(cats)

    q_labels = ['A', 'B', 'C', 'D']

    extra_qs = []
    count = 0
    #print(len(overlap)/len(cats))
    for x in new_q_data:
        if x['answer'] in overlap:
            sents = x['text']

            if 'et al.' in sents:
                #print(str(count) + ': ' + sents)
                sents = sents.replace('et al.', 'et al')
                #print(sents)

            
            
            sents = sent_tokenize(sents)
            for s in sents:
                count += 1
                y = dict()
                y['id'] = count
                y['question'] = s
                y['answer'] = ' '.join(x['answer'].split('_'))

                ans = [y['answer']]
                i = 0
                while i < 3:
                    n = random.randrange(len(cats))
                    c = ' '.join(cats[n].split('_'))

                    if (c != y['answer']) and (c not in ans):
                        ans.append(c)
                        i += 1

                random.shuffle(ans)

                for i in range(len(q_labels)):
                    key = 'answer' + q_labels[i]
                    y[key] = ans[i]

                    if ans[i] == y['answer']:
                        y['correctAnswer'] = q_labels[i]
                
                extra_qs.append(y)

    #print(extra_qs)
    #print(len(extra_qs))

    corpus_file = open("data/extra_corpus.txt", 'w')
    sp = sentence_parser()
    
    for x in extra_qs:
        sent = x['question']
        answer = x['answer']

        sent = sp.sub_pronoun(sent, answer)
        sent = sp.parse_sentence(sent)

        corpus_file.write(' '.join(sent) + '\n')

    corpus_file.close()

    all_keys = ["id", "question", "correctAnswer", "answerA", "answerB", "answerC", "answerD"]
    o = DictWriter(open("data/extra_train.csv", 'w'), all_keys, extrasaction="ignore")
    o.writeheader()
    for x in extra_qs:
        o.writerow(x)

    
