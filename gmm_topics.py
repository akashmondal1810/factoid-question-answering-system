# This is for Gaussian Mixture Model, to show probability within each clsuter
import os
import numpy as np
from gensim.models import Word2Vec
from word2vec_gen import *
import sklearn
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from csv import DictReader, DictWriter

# inputs: word2vec from wiki corpus data
# outputs: file of word, ordered clsuter, probs, centroids, varaince
# in one csv file

def word_clusters(out,vocab,probs,means,covs):
    # create output dictionary of words, order cluster idx, probs, centroids, (not saving varaince atm)
    word_clus = [dict() for x in range(len(vocab))]
    
    for ii in range(len(vocab)):
        word = vocab[ii]
        clus_prob = probs[ii]
        top_id = list()
        for idx in reversed(np.argsort(clus_prob)):
            top_id.append(idx)
        word_clus[ii] = {'word':word,'topicid':top_id,'topicval':list(clus_prob[top_id])} # 'clusmeans':means[top_id]};
        # excluding variances now - is it needed?
    
    keys = word_clus[0].keys()
 #  print(keys)
    out_file = out + '.csv' # for dictionary
    with open(out_file,'w',newline='',encoding='utf-8') as output: # trying not wb
        f = DictWriter(output,keys)
        f.writeheader()
        f.writerows(word_clus)
    
    out2 = out + '_means.txt' # for text file of cluster means
    np.savetxt(out2,means,delimiter=",") # each row is a cluster mean, row idx is clsuter idx (so not ordered)
    # use this to grab only cluster means needed, saves memory

## MAIN
if __name__ == "__main__":
# read in word2vec stuff
    print("Training word2vec...")

    filepath = "data/word2vec.dat"
    if os.path.exists(filepath):
        model = w2v(load_data=True, old_data=filepath)
        print("Existing data loaded.")
    else:
        model = w2v(data_path='data/wiki_corpus.txt')
        model.model.save(filepath)
        print("New word2vec data created.")

# read in vocab from word2vec set
    vocab = model.vocab #()   
    #print(vocab) # so voacb is all word of w2v model
    train = [model.wv(x) for x in vocab]
    #print(train[0])
   # print(len(vocab))
   # so vocab and train of same length, train is w2v represnetations


# from this, run on vectors for GMM
# LOOK UP GMM STUFF ON THEIR HELP SITE
    ncom = 20
    gmm = GaussianMixture(n_components=ncom,max_iter=500) # all other options should be defaults
    gmm.fit(train)
    weights = gmm.weights_ # these are weights of GMM
#    print(len(weights))
    means = gmm.means_ # these are means of GMM clusters
#    print(len(means))
    probs = gmm.predict_proba(train)
    covs = gmm.covariances_
    print("done first step")

# write to a csv file
    out_file = 'data/gmm_results_'+str(ncom)#.csv'
    word_clusters(out_file,vocab,probs,means,covs)


    ncom = 25
    gmm = BayesianGaussianMixture(n_components=ncom,max_iter=500) # they could potentially zero out some compoennts 
    gmm.fit(train)
    means = gmm.means_
    probs = gmm.predict_proba(train)
    covs = gmm.covariances_
    out_file = 'data/gmm_results_var_'+str(ncom)#.csv'
    word_clusters(out_file,vocab,probs,means,covs)


