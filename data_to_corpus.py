# Processes sentences from training data:
#   - removes stopwords
#   - stems words

# Also creates a file listing all words in the vocabulary and their
# frequency in the corpus. (To be changed.)


from csv import DictReader, DictWriter
from collections import defaultdict

from numpy import array
import numpy as np

import string

from nltk.tokenize import word_tokenize
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



class sentence_parser():

    def __init__(self):
        self.stemmer = SnowballStemmer("english")
#        self.stemmer = PorterStemmer()
        self.exclude = set(stopwords.words('english'))
        #punc = ["'", "''", '(', ')', ',', '.', '...', ';', ':', '[', ']', '``']
        punc = []

        self.exclude = self.exclude.union(set(punc))
        
        #override = ['i', 'y', 's', 'd', 'o']
        override = []
        for w in override:
            self.exclude.remove(w)

        self.pronouns = ['this', 'these', 'it', 'they', 'one', 'he', 'she', 'him',
                'its', 'his', 'her', 'their',
                'ones', 'here', 'type', 'them']


    # Given a sentence string, tokenize and stem, removing stopwords and
    # certain punctuation.
    def parse_sentence(self, sent):
        sent = word_tokenize(sent.lower())

        words_filtered = []

        for w in sent:
            if w not in self.exclude:
                words_filtered.append(w)

        return [self.stemmer.stem(w) for w in words_filtered]


    # Locates a (single) pronoun in the sentence, substituting the correct
    # answer into that location.
    def sub_pronoun(self, sent, word):
        sent = word_tokenize(sent)
        
        overlap = sorted(set(self.pronouns).intersection(sent),
                         key=lambda x:self.pronouns.index(x))

        if len(overlap) == 0:
            sent.insert(0,word)

        else:
            kw = overlap[0]
            ind = sent.index(kw)
            sent[ind] = word

        return ' '.join(sent)

    def wiki_sub(self, sent, page):
        sent = word_tokenize(remove_punctuation(sent))
        page_tok = word_tokenize(remove_punctuation(page))

        if not (set(sent) & set(page_tok)):
            
            overlap = sorted(set(self.pronouns).intersection(sent),
                             key=lambda x:self.pronouns.index(x))

            if len(overlap) == 0:
                sent.insert(0,page)

            else:
                kw = overlap[0]
                ind = sent.index(kw)
                sent[ind] = page

        return ' '.join(sent)


def remove_punctuation(text):
    return re.sub('[\W_]+', ' ', text, flags=re.UNICODE)


if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("data/sci_train.csv", 'r')))
    test = list(DictReader(open("data/sci_test.csv", 'r')))

    corpus_file = open("data/corpus.txt", 'w')
    vocab_file = open("data/vocab.csv", 'w')
    cat_file = open("data/categories.csv", 'w')

##    print(len(train))
##    print(len(test))


    sp = sentence_parser()
    
    #print(stop_words)

    vocab_count = defaultdict(lambda: defaultdict(int))
    categories = defaultdict(int)

    vocab = set()
    all_cats = set()
    ans_words = set()
    track = {'s':0, 'f':0}
    

    for x in train:
        
        sent = x[kQUESTION].lower()
        if len(sent) < 5 or 'tiebreaker' in sent:
            continue

        correct = 'answer' + x[kANSWER]
        sent = sp.sub_pronoun(sent, x[correct])

        sent = sp.parse_sentence(sent)

        # write processed sentence
        corpus_file.write(' '.join(sent) + '\n')

        all_cats.add(x[correct])
        categories[x[correct]] += 1

        ans = sp.parse_sentence(x[correct])

        for w in sent:
            vocab.add(w)
            vocab_count[w]['total'] += 1
            vocab_count[w][correct] += 1

        for w in ans:
            ans_words.add(w)



##    ans_words = list(ans_words)
##    ans_words.sort()
##    
##    for w in ans_words:
##        if w not in vocab:
##            track['f'] += 1
##            print(w + ' not in vocab')
##        else:
##            track['s'] += 1
##
##
##    print("Success: " + str(track['s']))
##    print("Fail: " + str(track['f']))

    vocab = list(vocab)
    vocab.sort()

    all_cats = list(all_cats)
    all_cats.sort()

    
    # write vocab to file
    for w in vocab:
        vocab_file.write(w + ',' + str(vocab_count[w]['total']) + '\n')

    for cat in all_cats:
        cat_file.write(cat + ',' + str(categories[cat]) + '\n')

    vocab_file.close()
    corpus_file.close()
    cat_file.close()
