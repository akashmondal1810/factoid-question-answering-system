## NMF, LDA portion of CMSC 726 Final Project


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # only for now, we should have word2vec
from sklearn.decomposition import NMF, LatentDirichletAllocation
import time
from scipy.sparse import csr_matrix
import nimfa
from word2vec_gen import *
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

def export_topics(out_file,topics,feature_names, n_top_words,ii_max=20):
    output = open(out_file, 'w',encoding='utf-8')
    for top_id in range(topics.shape[0]):
        output.write("==========\t%d\t==========\n" % (top_id))
        topic = topics[top_id, ]

        ii = 0
        for word_id in reversed(np.argsort(topic)):
            ii += 1
          #  print(topic)
            output.write("%s\t%g\n" % (feature_names[word_id],topic[word_id]))
            if ii_max > 0 and ii >= ii_max:
                break
            
    output.close()
#    
def word_topics(out_file,topics,feature_names,top_num=25):
    # first, need to make a list of dictionaries
    word_topics = [dict() for x in range(len(feature_names))]

    # loop through all words - order of feat_names and count_vec the same
    for ii in range(len(feature_names)):
        word = feature_names[ii]
        topic_vals = topics[:,ii] 
        # these should be indexces of the vector
        top_id = list()
        for idx in reversed(np.argsort(topic_vals)):
            top_id.append(idx)
        #print(top_id)
        # adding to the keys
        word_topics[ii] = {'word':word,'topicid':top_id,'topicval':list(topic_vals[top_id])}

 #  print(keys)
    with open(out_file,'w',newline='',encoding='utf-8') as output: # trying not wb
        f = DictWriter(output,keys)
        f.writeheader()
        f.writerows(word_topics)# use the rwiterows since you assume dict is a lsit of dicts)

           
    
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
## USING WIKI CORPUS
    inputfiletrain = "data/wiki_corpus.txt"
    #inputvocab = "data/vocab.txt"

    train = list((open(inputfiletrain,'r',encoding='utf-8')))
    #vocab = list((open(inputvocab,'r',encoding='utf-8')))

    print(len(train))
    N = len(train)

    feat = Featurizer()    
    x_train = feat.train_feature(x for x in train)
    
    #print(x_train[0])

    #### MAIN PARAMETER ##########
    n_topics = 20 # for now
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
    # current set-up --> no Elastic Net params
    nmf = NMF(n_components=n_topics, tol=1e-6,max_iter=500, alpha=0.0, l1_ratio=0.0).fit(x_train)
   #print_top_words(nmf, feat_names, 10)
    exposures = nmf.fit_transform(x_train)
    topics = nmf.components_
    print(topics.shape) 
    print()
#
# ## TRYING TO SAVE TO FILES NOW, do topics_per_sent in logreg file
    np.savetxt("topics_nmf_noreg.csv",topics,delimiter=",")
    np.savetxt("exposures_nmf_noreg.csv",exposures,delimiter=",")
    # calling topic - top words function
    export_topics("nmf_noreg_topicwords.txt",topics,feat_names,20)
    word_topics("nmf_noreg_words.csv",topics,feat_names,25)
    ## WRITING TO FILE WITH TOP WORDS IN TOPICS, ALONG WITH THE VALUE IN BETA/TOPICS VECT
    print("#\n Done NMF, sklearn, no EN \n #")
    
    
    nmf = NMF(n_components=n_topics, tol=1e-6,max_iter=500, alpha=0.9, l1_ratio=1.0).fit(x_train)
#    print_top_words(nmf, feat_names, 10)
    exposures = nmf.fit_transform(x_train)
    topics = nmf.components_
 
    np.savetxt("topics_nmf_lasso.csv",topics,delimiter=",")
    np.savetxt("exposures_nmf_lasso.csv",exposures,delimiter=",")
    export_topics("nmf_lasso_topicwords.txt",topics,feat_names,20)
    word_topics("nmf_lasso_words.csv",topics,feat_names,25)
    print("#\n Done NMF, sklearn, LASSO, a lot of regualrization\n #")
#    
#
    nmf = NMF(n_components=n_topics, tol=1e-6,max_iter=500, alpha=0.9, l1_ratio=0.0).fit(x_train)
    #print_top_words(nmf, feat_names, 20)
    exposures = nmf.fit_transform(x_train)
    topics = nmf.components_
    np.savetxt("topics_nmf_ridge.csv",topics,delimiter=",")
    np.savetxt("exposures_nmf_ridge.csv",exposures,delimiter=",")
    export_topics("nmf_ridge_topicwords.txt",topics,feat_names,20)
    word_topics("nmf_rdge_words.csv",topics,feat_names,25)
    print("#\n Done NMF, sklearn, RIDGE, a lot of regualrization\n #")
    
#        
    nmf = NMF(n_components=n_topics, tol=1e-6,max_iter=500, alpha=0.9, l1_ratio=0.5).fit(x_train)
#    print_top_words(nmf, feat_names, 20)
    exposures = nmf.fit_transform(x_train)
    topics = nmf.components_
    np.savetxt("topics_nmf_en.csv",topics,delimiter=",")
    np.savetxt("exposures_nmf_en.csv",exposures,delimiter=",")
    export_topics("nmf_en_topicwords.txt",topics,feat_names,20)
    word_topics("nmf_en_words.csv",topics,feat_names,25)
    print("#\n Done NMF, sklearn, EN = even, less regualrization\n #")
    
#######################################################################################
#
####################################################################################
#
#    ## LDA part
#    # n_topics --> number of topics, K
#    # doc_topic_prior --> alpha parameter of document-topic distribution (assume same for each topic,
#        # considering we assume topics evenly distributed)
#    #   WANT THIS TO BE SMALL (SO SPARSE, since we assuem small # topics, or really 1, for each sentence)
#    
#    #  topic_word_prior --> prior of beta dsit, or word dist for each topic
#    #   both defaults are 1/n_topics, which are fairly sparse. But will be important to tune
#    # maybe we should assume doc_topic more sparse than topic_word?
#    # max_iter --> default 10, set to 100                      
#    # batch_size --> used only for online, default 128, number of documents for each onlie step, not totally stochastic
#    
#    # max_change_tol --> default 1e-3, stopping crit for updaing doc-tpic dist
#    # max_doc_update_iter --> if ou stop in ceratain iter instead of tolerance, default 100
#    
    lda = LatentDirichletAllocation(n_topics=n_topics, doc_topic_prior=0.05, topic_word_prior = 0.05, learning_method = 'online', max_iter=100, mean_change_tol = 1e-5, max_doc_update_iter = 500)
    lda.fit(x_train)
#    print_top_words(lda, feat_names, 20) # 20 top words
    exposures = lda.transform(x_train)
    topics = lda.components_

    np.savetxt("topics_lda_sparse.csv",topics,delimiter=",")
    np.savetxt("exposures_lda_sparse.csv",exposures,delimiter=",")
    export_topics("lda_sparse_topicwords.txt",topics,feat_names,20)
    word_topics("lda_sparse_words.csv",topics,feat_names,25)
    print("# \n Done LDA part, strict sparsity of prior \n #")
    
    lda = LatentDirichletAllocation(n_topics=n_topics, doc_topic_prior=0.25, topic_word_prior = 0.05, learning_method = 'online', max_iter=100, mean_change_tol = 1e-5, max_doc_update_iter = 500)
    lda.fit(x_train)
#    print_top_words(lda, feat_names, 20) # 20 top words
    exposures = lda.transform(x_train)
    topics = lda.components_

    np.savetxt("topics_lda_non_sparse.csv",topics,delimiter=",")
    np.savetxt("exposures_lda_non_sparse.csv",exposures,delimiter=",")
    export_topics("lda_non_topicwords.txt",topics,feat_names,20)
    word_topics("lda_non_words.csv",topics,feat_names,25)
    print("# \n Done LDA part, little sparsity of prior \n #")
    
#    ######################################################
#    #### KL divergence tries
#    # NMF KL- NIMFA style
##
    klnmf = nimfa.Nmf(np.transpose(x_train),max_iter=100,n_run=1, rank = n_topics, update='divergence', objective = 'div')
    klnmf_fit = klnmf()
    
    # here, W is M x K, H is K x N. W is basis, H is abndances, get transposes
    topics = klnmf_fit.basis()
    print(type(topics))
    exposures = klnmf_fit.coef()
    
    topics.todense() # making scipy sparse to dense
    exposures.todense()
    
    topics = topics.toarray()
    exposures = exposures.toarray()
    # to make them scipy arrays

   # in order to match matrix dimesnions of previous methods
    topics = np.transpose(topics)
    exposures = np.transpose(exposures)
    n_top_words = 10

    np.savetxt("topics_klnmf.csv",topics,delimiter=",")
    np.savetxt("exposures_klnmf.csv",exposures,delimiter=",")
    export_topics("nimfa_topicwords.txt",topics,feat_names,20)
    word_topics("nimfa_kl_words.csv",topics,feat_names,20)
    print(" # \n Done nimfa, Kl divergence \n #")
