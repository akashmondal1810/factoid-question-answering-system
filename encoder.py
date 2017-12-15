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
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Add, Concatenate
from keras.datasets import imdb
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

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
    x_test, y_test = formVector(test)

    input_texts = x_train
    target_texts = y_train
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    
    # print('input len',max_encoder_seq_length )
    encoder_inputs = Input(shape=(1, 250))

    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    # print(encoder_outputs)
    encoder_states = [state_h, state_c]
    concat_states = Concatenate(axis=-1)([state_h, state_c])
    print(concat_states)
    sess = tf.Session()
    with sess.as_default():        
        print(type(concat_states.eval()))
        print("x_train_before", x_train)
    # x_train = np.array(concat_states)
    
    y_train = np.array(y_train)	
    model = Model(encoder_inputs, encoder_outputs)
    # print("x_train", x_train)
    # print("y_train", y_train[1])
    # Run training
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)
    # lr = LogisticRegression(C=1, penalty='l2', fit_intercept=True)
    # print(x_train.shape)
    # lr.fit(x_train, y_train)
    # sess = tf.Session()
    # with sess.as_default():        
    print(type(concat_states.eval()))
    print("x_train_before", x_train)
	
	
	
    
    # print("x train val:",x_train[1])