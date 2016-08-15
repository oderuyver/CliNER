
from collections import defaultdict
import os
import sys
from time import localtime, strftime

import keras_ml
from documents import labels as tag2id, id2tag
from tools import flatten
from tools import pickle_dump, load_pickled_obj



def print_files(f, file_names):
    COLUMNS = 4
    file_names = sorted(file_names)
    start = 0
    for row in range(len(file_names)/COLUMNS + 1):
        print >>f,'\t\t',
        for featname in file_names[start:start+COLUMNS]:
            print >>f, '%-15s' % featname,
        print >>f, ''
        start += COLUMNS


def print_vec(f, label, vec):
    COLUMNS = 7
    start = 0
    print >>f, '\t%s: ' % label
    if type(vec) != type([]):
        vec = vec.tolist()
    for row in range(len(vec)/COLUMNS + 1):
        print >>f,'\t   ',
        for featname in vec[start:start+COLUMNS]:
            print >>f, '%7.3f' % featname,
        print >>f, ''
        start += COLUMNS


class Galen:

    def log(self, filename, logfile):
        with open(logfile, 'a') as f:
            print >>f, ''
            print >>f, '-'*40
            print >>f, ''
            print >>f, 'model         : ', filename
            print >>f, 'training began: ', self._time_train_begin
            print >>f, 'training ended: ', self._time_train_end
            print >>f, ''
            print >>f, 'scores'
            print_vec(f, 'precision', self._score['precision'])
            print_vec(f, 'recall'   , self._score['precision'])
            print_vec(f, 'f1'       , self._score['precision'])
            for label,vec in self._score['history'  ].items():
                print_vec(f, label, vec)
            print >>f, ''
            print >>f, 'Training Files'
            if self._training_files:
                print_files(f, self._training_files)
                print >>f, ''
            print >>f, '-'*40
            print >>f, ''


    def __init__(self):
        self._clf            = None
        self._vocab          = None
        self._training_files = None


    def fit_from_notes(self, notes):
        """
        Galen::train()

        Purpose: Train a Machine Learning model on annotated data

        @param notes. A list of Document objects (containing text and annotations)
        @return       None
        """
        # Extract formatted data
        tokenized_sents  = flatten([n.getTokenizedSentences() for n in notes])
        labels           = flatten([n.getTokenLabels()        for n in notes])

        # Call the internal method
        self.fit(tokenized_sents, labels)

        self._training_files = [ n.getName() for n in notes ]



    def fit(self, tokenized_sents, tags):
        '''
        Fit a Galen model to predict concept span tags for given sentences.

        Arguments:
            tokenized_sents: A list of sentences, where each sentence is tokenized
                            into words
            tags: Parallel to `tokenized_sents`, 7-way labels for concept spans
        '''

        # metadata
        self._time_train_begin = strftime("%Y-%m-%d %H:%M:%S", localtime())

        # train classifier
        voc, clf, dev_score = generic_train('all', tokenized_sents, tags)
        self._vocab = voc
        self._clf   = clf
        self._score = dev_score

        # metadata
        self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())


    def predict(self, note):
        """
        Galen::predict()

        Purpose: Predict concept annotations for a given note

        @param note. A Document object (containing text and annotations)
        @return      <list> of predictions
        """
        # Extract formatted data
        tokenized_sents  = note.getTokenizedSentences()

        # Predict labels for prose
        num_pred = generic_predict('all'                   ,
                                   tokenized_sents         ,
                                   vocab    = self._vocab  ,
                                   clf      = self._clf    )
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

    print '\tvectorizing words', p_or_n

    # build vocabulary of words
    vocab = {}
    for sent in tokenized_sents:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab) + 1
    vocab['oov'] = len(vocab)

    # vectorize tokenized sentences
    X_seq_ids = []
    for sent in tokenized_sents:
        id_seq = [ (vocab[w] if w in vocab else vocab['oov']) for w in sent ]
        X_seq_ids.append(id_seq)

    # Flatten and vectorize IOB labels
    Y_labels = [ [tag2id[y] for y in y_seq] for y_seq in iob_nested_labels ]

    print '\ttraining classifiers', p_or_n

    # train using lstm
    clf, dev_score  = keras_ml.train(X_seq_ids, Y_labels, tag2id, W=None)

    return vocab, clf, dev_score



def generic_predict(p_or_n, tokenized_sents, vocab, clf):
    # If nothing to predict, skip actual prediction
    if len(tokenized_sents) == 0:
        print '\tnothing to predict ' + p_or_n
        return []

    print '\tvectorizing words ' + p_or_n

    # vectorize tokenized sentences
    X_seq_ids = []
    for sent in tokenized_sents:
        id_seq = []
        for w in sent:
            if w in vocab:
                id_seq.append(vocab[w])
            else:
                id_seq.append(vocab['oov'])
        X_seq_ids.append(id_seq)

    print '\tpredicting  labels ' + p_or_n

    # Predict labels
    predictions = keras_ml.predict(clf, X_seq_ids)

    # Format labels from output
    return predictions

