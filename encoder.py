from __future__ import print_function

from keras.models import Model
from keras.layers import Input
from csv import DictReader, DictWriter
import numpy as np
import sklearn
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from word2vec_gen import *

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'
kID_FIELD = 'id'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.


def formVector(data):
    # lablelist = ["A", "B", "C", "D"]
    x_train = []
    y_train = []
    for x in data:
        
        interlist = []
        for c in x[kTEXT_FIELD]:

            q = model.sv(c)

            interlist.append(q)

        v = interlist
        y = np.hstack((model.sv(x[kA])))
        y = np.hstack((y, model.sv(x[kB])))
        y = np.hstack((y, model.sv(x[kC])))
        y = np.hstack((y, model.sv(x[kD])))      

       
        x_train.append(v)
        y_train.append(y)

    
    return x_train, y_train

if __name__ == "__main__":

    n = 0.7  # train validation split
    train = list(DictReader(open("data/filtered_train.csv", 'r')))   
    train = shuffle(train)
    print("Total length: ", len(train))
    test = train[-int(len(train) * (1.0 - n)):]
    train = train[:int(len(train) * n)]

    model = w2v()

    x_train, y_train = formVector(train)
    # x_test, y_test = formVector(test)

    print('x_train type :',type(x_train),"x_train Value:",x_train[1])

    # karan = model.sv('science is bi tch')
    # karan = 'sci is bt ch'
    # words = karan.split()  
    # f = type(words)
    # print('type:',f)

    # for i in words:
    # 	print(i)


    # print('karan val:',words)




    