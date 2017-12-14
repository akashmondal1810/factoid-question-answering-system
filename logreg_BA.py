from csv import DictReader, DictWriter
import numpy as np
import sklearn

import argparse

from numpy import array
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
from word2vec_gen import *
from gensim.models import Word2Vec

from data_to_corpus import sentence_parser

import json
from collections import defaultdict

import os.path

kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'
kID_FIELD = 'id'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'


class sgd_classifier():

    def __init__(self, w2v_model, sv, loss='log', penalty='l2',kernel='linear',C=1,
                 use_topics=False,topics=dict(),N_tops=0):

        if kernel=='linear':
            self.lr = SGDClassifier(loss=loss, penalty=penalty,max_iter=1000,tol=10**-3)
        else:
            self.lr = SVC(C=C,kernel=kernel)
            
        self.model = w2v_model
        self.sv = sv

        self.use_topics=use_topics
        self.topics=topics
        self.topic_voc=list(topics.keys())

        #print(self.topics)
        #print(self.topics['k-sub-5'])
        
        self.N_tops=N_tops

        self.sp = sentence_parser()


    def form_vector(self, question, answer):
        v = np.hstack((self.sv(question), self.sv(answer)))

        if self.use_topics:
            tv = np.hstack((self.get_topic_vector(question), self.get_topic_vector(answer)))
            v = np.hstack((v, tv))
        
        return v
        

    def form_all_vectors(self, data):
        label_list = ["A", "B", "C", "D"]
        x_train = []
        y_train = []
        for x in data:
            for label in label_list:
                alabel = 'answer' + label
                v = self.form_vector(x[kTEXT_FIELD], x[alabel])
                x_train.append(v)

                if label == x['correctAnswer']:
                    y_train.append(1)
                else:
                    y_train.append(0)

        return x_train, y_train


    def get_topic_vector(self, sent):
        v = np.zeros(self.N_tops)

        for w in self.sp.parse_sentence(sent):
            if w not in self.topics.keys():
                continue
            
            #print(w)
            #print(self.topics[w])
            v += self.topics[w]['topicvals']

        nv = np.linalg.norm(v)
        
        if nv > 1e-5:
            v /= nv
        
        return v


    def train(self, x_train, y_train):
        self.lr.fit(x_train, y_train)

    def predict(self, x_data):
        predictions = []

        true_ind = np.argmax(self.lr.classes_)
        #print(self.lr.classes_)
        #print(true_ind)
        
        for i in range(0,len(x_data),4):
            X = x_data[i:i+4]
            p = self.lr.decision_function(X)

            if true_ind == 0:
                p = -p

            predictions.append(np.argmax(p))

        return predictions
        

    def test(self, x_data, y_data):
        success = 0
        failure = 0

        predictions = self.predict(x_data)
        
        for i in range(0,len(predictions)):
            ans = predictions[i]

            if y_data[4*i+ans] == 1:
                success += 1
            else:
                failure += 1

        return [success, failure]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--loss', default='log', type=str,
                        help="Loss function")
    parser.add_argument('--penalty', default='l2', type=str,
                        help="Regularization punalty")
    parser.add_argument('--kernel', default='linear', type=str,
                        help="SVM kernel")
    parser.add_argument('--C', default=1.0, type=float,
                        help="Slack penalty")
    parser.add_argument('--topics', default='gmm', type=str,
                        help="Algorithm that generated topics")

    flags = parser.parse_args()

    n = 0.8  # train validation split

    train = list(DictReader(open("data/filtered_train.csv", 'r')))
    extra_train = list(DictReader(open("data/extra_train.csv", 'r')))

    use_topics = True

    if flags.topics == "gmm":
        topic_file = "data/gmm_results_20.csv"
    elif flags.topics == "gmm_var":
        topic_file = "data/gmm_results_var_25.csv"
    elif flags.topics == "lda":
        topic_file = "data/lda_sparse_words.csv"
    elif flags.topics == "nimfa":
        topic_file = "data/nimfa_kl_words.csv"
    else:
        use_topics = False


    topics = defaultdict(dict)
    N_tops = 0

    if use_topics:
        topics_raw = list(DictReader(open(topic_file, 'r')))

        for x in topics_raw:
            word = x['word']
            tops = json.loads(x['topicid'])
            N_tops = len(tops)
            weights = json.loads(x['topicval'])

            tops, weights = (list(t) for t in zip(*sorted(zip(tops, weights))))


            topics[word]['topics'] = tops
            topics[word]['topicvals'] = weights


            #print(topics[word])

        
    train = shuffle(train)
    extra_train = shuffle(extra_train)
    #train=extra_train
    #print(train[0])

    ind = int(len(train)*n)
    test = train[-(len(train) - ind):]
    train = train[:ind]

    train_len = len(train)
    test_len = len(test)
    
    print("Train size: ", train_len)
    print("Test size: ", test_len)

    print("Training word2vec...")

    filepath = "data/word2vec.dat"
    if os.path.exists(filepath):
        model = w2v(load_data=True, old_data=filepath)
        print("Existing data loaded.")
    else:
        model = w2v(data_path='data/wiki_corpus.txt')
        model.model.save(filepath)
        print("New word2vec data created.")


    lr = sgd_classifier(model, model.sv, loss=flags.loss, penalty=flags.penalty,
                        kernel=flags.kernel, C=flags.C,
                        use_topics=use_topics,topics=topics,N_tops=N_tops)

    print("Forming vectors...")
    x_train, y_train = lr.form_all_vectors(train)
    x_test, y_test = lr.form_all_vectors(test)

    print("Training classifier...")
    lr.train(x_train, y_train)
    [success, failure] = lr.test(x_train, y_train)
    accuracy_train = success / (success + failure) * 100

    print("Training accuracy:")
    print(accuracy_train)

    [success, failure] = lr.test(x_test, y_test)
    accuracy_val = success / (success + failure) * 100
    
    print("Validation accuracy:")
    print(accuracy_val)

    predictions = lr.predict(x_test)

    # print(predictions[:10])

##    o = DictWriter(open("data/testpredictions.csv", 'w'), ["id", "correctAnswer"])
##    o.writeheader()
##    for ii, pp in zip([x['id'] for x in test], predictions):
##        d = {'id': ii, 'correctAnswer': pp}
##        o.writerow(d)
