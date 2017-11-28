## NMF, LDA portion of CMSC 726 Final Project


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # only for now, we should have word2vec
from sklearn.decomposition import NMF, LatentDirichletAllocation
import time
from scipy.sparse import csr_matrix
import nimfa


from csv import DictReader, DictWriter

kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'
kID_FIELD = 'id'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'


def topics_per_sent(exposures,numtop):
    topics_per_sentence = dict()
    for ii in range(exposures.shape[0]): # for all sentences
        sentexp = exposures[ii]
        if ii == 100 or ii == 2000 or ii == 734:
            print(sentexp) # MEANT TO TEST HOW SPARSE EXPOSURES ARE

        top_topics = sentexp.argsort()[:-numtop-1:-1]
        topics_per_sentence[ii] = topics[top_topics]
        
    return topics_per_sentence
    
    
# from online help to see top features
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer = 'word')

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)
    
    def feature_names(self):
        return self.vectorizer.get_feature_names()

# main function
if __name__ == "__main__":
#    inputfiletrain = "data/sci_train.csv"
#    inputfiletest = "data/sci_test.csv"
    inputfiletrain = "data/corpus.txt"
# READING IN the sentences, with embeddings, should be 5441   

    train = list((open(inputfiletrain,'r',encoding='utf-8')))
    print(len(train))
    N = len(train)

    feat = Featurizer()    
    x_train = feat.train_feature(x for x in train)
    #print(x_train[0])

    #### MAIN PARAMETER ##########
    n_topics = 10 # for now
    ##################################
    feat_names = feat.feature_names()
  
    ## PARAMETERS FOR NMF
#    # n_components = # of topics guessed
#    # tol - default is 1e-4, doing mine for greater granuarilty of solutin, run time is OK
#    # max_iter - default 200
#    # random_state - if we want to sed with 1,2,3, etc
#    # alpha - the constant coefficient, applied to both W and H
#    #   scale how much regularization is done, 0.1 means not much, 1 means a lot
#    # l1_ratio - what is typically 'alpha' in EN discussion, 0 means ridge, 1 means LASSO
#    # current set-up --> no Elastic Net params
    nmf = NMF(n_components=n_topics, tol=1e-6,max_iter=500, alpha=0.0, l1_ratio=0.0).fit(x_train)
    print_top_words(nmf, feat_names, 20)
    exposures = nmf.fit_transform(x_train)
    topics = nmf.components_
    print(topics.shape) 
    print()

 ## TRYING TO SAVE TO FILES NOW, do topics_per_sent in logreg file
    np.savetxt("atopics_nmf_noreg.csv",topics,delimiter=",")
    np.savetxt("aexposures_nmf_noreg.csv",exposures,delimiter=",")
    print("#\n Done NMF, sklearn, no EN \n #")
    
    
    nmf = NMF(n_components=n_topics, tol=1e-6,max_iter=500, alpha=0.9, l1_ratio=1.0).fit(x_train)
    print_top_words(nmf, feat_names, 20)
    exposures = nmf.fit_transform(x_train)
    topics = nmf.components_
 
    np.savetxt("atopics_nmf_lasso.csv",topics,delimiter=",")
    np.savetxt("aexposures_nmf_lasso.csv",exposures,delimiter=",")
    print("#\n Done NMF, sklearn, LASSO, a lot of regualrization\n #")
#    
#
    nmf = NMF(n_components=n_topics, tol=1e-6,max_iter=500, alpha=0.9, l1_ratio=0.0).fit(x_train)
    print_top_words(nmf, feat_names, 20)
    exposures = nmf.fit_transform(x_train)
    topics = nmf.components_
    np.savetxt("atopics_nmf_ridge.csv",topics,delimiter=",")
    np.savetxt("aexposures_nmf_ridge.csv",exposures,delimiter=",")
    print("#\n Done NMF, sklearn, RIDGE, a lot of regualrization\n #")
    
#        
    nmf = NMF(n_components=n_topics, tol=1e-6,max_iter=500, alpha=0.9, l1_ratio=0.5).fit(x_train)
    print_top_words(nmf, feat_names, 20)
    exposures = nmf.fit_transform(x_train)
    topics = nmf.components_
    np.savetxt("atopics_nmf_en.csv",topics,delimiter=",")
    np.savetxt("aexposures_nmf_en.csv",exposures,delimiter=",")
    print("#\n Done NMF, sklearn, EN = even, less regualrization\n #")
    
######################################################################################

###################################################################################

    ## LDA part
    # n_topics --> number of topics, K
    # doc_topic_prior --> alpha parameter of document-topic distribution (assume same for each topic,
        # considering we assume topics evenly distributed)
    #   WANT THIS TO BE SMALL (SO SPARSE, since we assuem small # topics, or really 1, for each sentence)
    
    #  topic_word_prior --> prior of beta dsit, or word dist for each topic
    #   both defaults are 1/n_topics, which are fairly sparse. But will be important to tune
    # maybe we should assume doc_topic more sparse than topic_word?
    # max_iter --> default 10, set to 100                      
    # batch_size --> used only for online, default 128, number of documents for each onlie step, not totally stochastic
    
    # max_change_tol --> default 1e-3, stopping crit for updaing doc-tpic dist
    # max_doc_update_iter --> if ou stop in ceratain iter instead of tolerance, default 100
    
    lda = LatentDirichletAllocation(n_topics=n_topics, doc_topic_prior=0.05, topic_word_prior = 0.05, learning_method = 'online', max_iter=20, mean_change_tol = 1e-5, max_doc_update_iter = 500)
    lda.fit(x_train)
    print_top_words(lda, feat_names, 20) # 20 top words
    exposures = lda.transform(x_train)
    topics = lda.components_

    np.savetxt("atopics_lda_sparse.csv",topics,delimiter=",")
    np.savetxt("aexposures_lda_sparse.csv",exposures,delimiter=",")
    print("# \n Done LDA part, strict sparsity of prior \n #")
    
    lda = LatentDirichletAllocation(n_topics=n_topics, doc_topic_prior=0.25, topic_word_prior = 0.25, learning_method = 'online', max_iter=20, mean_change_tol = 1e-5, max_doc_update_iter = 500)
    lda.fit(x_train)
    print_top_words(lda, feat_names, 20) # 20 top words
    exposures = lda.transform(x_train)
    topics = lda.components_

    np.savetxt("atopics_lda_sparse.csv",topics,delimiter=",")
    np.savetxt("aexposures_lda_sparse.csv",exposures,delimiter=",")
    print("# \n Done LDA part, little sparsity of prior \n #")
    
    ######################################################
    #### KL divergence tries

   
    # KLdivNMF
#    from nmf_kl import KLdivNMF
#    # only random initializations
#    klnmf = KLdivNMF(n_components=10,tol=1e-6,max_iter=500)
#    klnmf.fit(x_train)
#    print_top_words(klnmf,feat_names,20)
#    print("# \n done KLdivNMF, without changing update \n #")

    # NMF KL- NIMFA style
    start_time = time.time()
    print("##\n##\n seeing tpye of x_train that is issue\n##\n##")

    klnmf = nimfa.Nmf(np.transpose(x_train),max_iter=500,n_run=5, rank = n_topics, update='divergence', objective = 'div')
    klnmf_fit = klnmf()
    print('-- %s seconds for NMF, 500 iters, divergence as update (longer name), div as obj --' % (time.time() - start_time))
    # here, W is M x K, H is K x N. W is basis, H is abndances, get transposes
    topics = klnmf_fit.basis()
    print(type(topics))
    exposures = klnmf_fit.coef()
    
    topics.todense() # making scipy sparse to dense
    exposures.todense()
    
    topics = topics.toarray()
    exposures = exposures.toarray()
    # to make them scipy arrays
   # print(type(topics))
   # in order to match matrix dimesnions of previous methods
    topics = np.transpose(topics)
    exposures = np.transpose(exposures)
    n_top_words = 10

    for topic_idx, topic in enumerate(topics):
        print("Topic #%d:" % topic_idx)
        #topicrow = topic[0].ravel()
        topicrow = topic.ravel()
        print(" ".join([feat_names[i]
                        for i in topicrow.argsort()[:-n_top_words - 1:-1]]))
    print()

    np.savetxt("atopics_klnmf.csv",topics,delimiter=",")
    np.savetxt("aexposures_klnmf.csv",exposures,delimiter=",")
    print(" # \n Done nimfa, Kl divergence \n #")

    
#    start_time = time.time()
#    klnmf = nimfa.Nmf(np.transpose(x_train),max_iter=500,n_run=1, rank = 10, update='euclidean', objective = 'fro')
#    klnmf_fit = klnmf()
#    print('-- %s seconds for NMF, 500 iters, euclidean as update, fro as obj --' % (time.time() - start_time))
#    # here, W is M x K, H is K x N. W is basis, H is abndances, et transposes
#    W = klnmf_fit.basis()
#    H = klnmf_fit.coef()
#    for topic_idx, topic in enumerate(np.transpose(W)):
#        print("Topic #%d:" % topic_idx)
#        topicrow = topic.toarray()[0].ravel()
#        print(" ".join([feat_names[i]
#                        for i in topicrow.argsort()[:-n_top_words - 1:-1]]))
#    print()
    print(" # \n Not writing this for nimfa, Euclidean \n #")