
import sys
import os
from collections import defaultdict
from time import localtime, strftime

import keras_ml

import numpy as np

from tools import flatten, save_list_structure, reconstruct_list
from tools import pickle_dump, load_pickled_obj

from note import labels as tag2id




# reverse this dict
id2tag = { v:k for k,v in tag2id.items() }



def print_features(f, label, feature_names):
    COLUMNS = 4
    feature_names = sorted(feature_names)
    print >>f, '\t    %s' % label
    start = 0
    for row in range(len(feature_names)/COLUMNS + 1):
        print >>f,'\t\t',
        for featname in feature_names[start:start+COLUMNS]:
            print >>f, '%-15s' % featname,
        print >>f, ''
        start += COLUMNS



class Model:

    @staticmethod
    def load(filename):
        model = load_pickled_obj(filename)
        model.filename = filename
        return model


    def serialize(self, filename, logfile=None):
        # Serialize the model
        pickle_dump(self, filename)

        # Describe training info?
        if logfile:
            #with open(logfile, 'a') as f:
            f = sys.stdout
            if True:
                print >>f, '-'*40
                print >>f, ''
                print >>f, 'model         : ', filename
                print >>f, 'training began: ', self._time_train_begin
                print >>f, 'training ended: ', self._time_train_end
                print >>f, ''
                print >>f, 'Training Files'
                print >>f, ''
                print_features(f, 'files', self._training_files)
                print >>f, ''
                print >>f, '-'*40



    def __init__(self):
        # Classifiers
        self._clf     = None
        self._vocab   = None



    def train(self, notes):
        """
        Model::train()

        Purpose: Train a Machine Learning model on annotated data

        @param notes. A list of Note objects (containing text and annotations)
        @return       None
        """

        # metadata
        self._time_train_begin = strftime("%Y-%m-%d %H:%M:%S", localtime())

        # Extract formatted data
        tokenized_sents  = flatten([n.getTokenizedSentences() for n in notes])
        labels           = flatten([n.getTokenLabels()        for n in notes])

        #tokenized_sents = tokenized_sents[15:20]
        #labels          =          labels[15:20]

        '''
        # experiment! Do POS tagging or something
        from nltk.corpus import brown
        nb_samples = 5000
        full_corpus = brown.tagged_sents(tagset='universal')
        corpus = list(full_corpus[0:nb_samples])

        tokenized_sents = []
        labels = []
        for tagged_sent in corpus:
            tokenized_sents.append( [ t[0].lower() for t in tagged_sent ] )
            labels.append(          [ t[1]         for t in tagged_sent ] )

        global tag2id
        tag2id = {}
        for tags in labels:
            for tag in tags:
                if tag not in tag2id:
                    tag2id[tag] = len(tag2id)
        # -- end experiment
        '''

        # train classifier
        voc, clf, vect = generic_train('all', tokenized_sents, labels)
        self._vocab = voc
        self._clf   = clf
        self._vect  = vect

        # metadata
        self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())
        self._training_files = [ n.getName() for n in notes ]



    def predict(self, note):
        """
        Model::predict()

        Purpose: Predict concept annotations for a given note

        @param note. A Note object (containing text and annotations)
        @return      <list> of Classification objects
        """

        # Extract formatted data
        tokenized_sents  = note.getTokenizedSentences()

        # Train classifiers for 1st pass and 2nd pass
        print '\textracting  features'

        # Predict labels for prose
        num_pred = generic_predict('all'                   ,
                                   tokenized_sents         ,
                                   vocab    = self._vocab  ,
                                   clf      = self._clf    ,
                                   dict2vec = self._vect   )
        iob_pred = [ [id2tag[p] for p in seq] for seq in num_pred ]

        return iob_pred




def generic_train(p_or_n, tokenized_sents, iob_nested_labels):
    '''
    generic_train()

    Purpose: Train that works for both prose and nonprose
    '''

    # Must have data to train on
    if len(tokenized_sents) == 0:
        raise Exception('Training must have %s training examples' % p_or_n)

    #tokenized_sents   =   tokenized_sents * 40
    #iob_nested_labels = iob_nested_labels * 40

    #print len(tokenized_sents)
    #exit()

    '''
    #####################################################
    from nltk.corpus import brown
    nb_brown_samples = 2000
    brown_corpus = brown.tagged_sents(tagset='universal')
    corpus = list(brown_corpus[0:nb_brown_samples])

    tokenized_sents   = []
    iob_nested_labels = []
    tags = set()
    for tagged_sent in corpus:
        for sent,pos_tags in tagged_sent:
            tokenized_sents.append(sent)
            iob_nested_labels.append(pos_tags)
            tags = tags | set(pos_tags)
    tag2id = { t:i for i,t in enumerate(tags) }
    id2tag = { v:k for k,v in tag2id.items() }
    #####################################################
    '''

    print '\tvectorizing words', p_or_n

    # which tokens to remove
    oov = create_oov(tokenized_sents, percent=.02)
    #oov = set()

    # build vocabulary of words
    vocab = {}
    for sent in tokenized_sents:
        for w in sent:
            #if w not in oov:
                if w not in vocab:
                    vocab[w] = len(vocab) + 1
    vocab['oov'] = len(vocab)

    #print 'len: ', len(tokenized_sents)
    #print 'V:   ', len(vocab)
    #exit()

    # vectorize tokenized sentences
    X_seq_ids = []
    for sent in tokenized_sents:
        id_seq = [ (vocab[w] if w in vocab else vocab['oov']) for w in sent ]
        X_seq_ids.append(id_seq)
    #X_seq_ids = [ [ vocab[w] for w in sent ] for sent in tokenized_sents ]
    dict2vec = None

    # Flatten and vectorize IOB labels
    Y_labels = [ [tag2id[y] for y in y_seq] for y_seq in iob_nested_labels ]

    print '\ttraining classifiers', p_or_n

    # Train classifier
    #print X_seq_ids
    #print Y_labels
    #exit()

    '''
    # load word2vec embeddings
    word2vec_file = '/data1/wboag/pet/galen/misc/mimic.word2vec'
    W_dict = load_word2vec(word2vec_file)
    dim = W_dict.values()[0].shape[0]

    # initialize embedding layer W with word2vec
    W = np.zeros( (len(vocab), dim) )
    for w,ind in vocab.items():
        if w in W_dict:
            W[ind,:] = W_dict[w]
        else:
            W[ind,:] = np.random.rand(dim)
    '''
    W = None

    # predict using lstm
    clf  = keras_ml.train(X_seq_ids, Y_labels, tag2id, W)

    return vocab, clf, dict2vec



def generic_predict(p_or_n, tokenized_sents, vocab, clf, dict2vec):

    # If nothing to predict, skip actual prediction
    if len(tokenized_sents) == 0:
        print '\tnothing to predict ' + p_or_n
        return []


    print '\tvectorizing words ' + p_or_n

    # vectorize tokenized sentences
    X_seq_ids = []
    oov_count = 0
    all_count = 0
    for sent in tokenized_sents:
        id_seq = []
        for w in sent:
            if w in vocab:
                id_seq.append(vocab[w])
            else:
                id_seq.append(vocab['oov'])
                #print 'oov: ', w
                oov_count += 1
            all_count += 1
        X_seq_ids.append(id_seq)

    '''
    print
    print 'OOV ratio: ', float(oov_count)/all_count
    print

    # unpack params for model
    lstm_model_str, input_dim, num_tags, maxlen = clf

    # dump serialized model out to file in order to load it
    with open('lstm_keras_weights', 'wb') as f:
        f.write(lstm_model_str)
    lstm_model = keras_ml.create_model(input_dim, num_tags, maxlen)
    lstm_model.load_weights('lstm_keras_weights')
    os.remove('lstm_keras_weights')

    print
    print '|V|: ', len(vocab)
    print

    vals = lstm_model.get_weights() # i,c,f,o
    '''

    print '\tpredicting  labels ' + p_or_n

    # Predict labels
    predictions = keras_ml.predict(clf, X_seq_ids)

    # Format labels from output
    return predictions


num_feats = 20




def create_oov(tokenized_sents, percent=.05):
    freqs = defaultdict(int)
    for sent in tokenized_sents:
        for w in sent:
            freqs[w] += 1
    total_words = sum(freqs.values())
    pdf = { w:float(f)/total_words for w,f in freqs.items() }
    sorted_pdf = sorted(pdf.items(), key=lambda t:t[1])

    # build cumulative distribution function
    total_mass = 0
    cdf = []
    for w,p in sorted_pdf:
        total_mass += p
        cdf.append( (w,total_mass) )

    # find the 5% mark for word frequencies
    for ind in range(len(cdf)):
        if cdf[ind][1] >= percent:
            break
    oov = set()
    for i in range(ind):
        oov.add( cdf[i][0] )

    return oov


def load_word2vec(W_file):
    W = {}
    with open(W_file, 'r') as f:
        for i,line in enumerate(f.readlines()[1:]):
            toks = line.split()

            word = toks[0]
            vec  = np.array(map(float,toks[1:]))

            #vec /= length(vec)

            W[word] = vec
    return W

