

import numpy as np
import os
import random
import time


from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge


# only load compile this model once per run (useful when predicting many times)
lstm_model = None


def train(X_ids, Y_ids, tag2id, W=None, epochs=5):
    # gotta beef it up sometimes
    #X_ids = X_ids * 5
    #Y_ids = Y_ids * 5

    # build model
    input_dim    = max(map(max, X_ids)) + 1
    maxlen       = max(map(len, X_ids))
    num_tags     = len(tag2id)
    lstm_model = create_bidirectional_lstm(input_dim, num_tags, maxlen,W=W)

    # turn each id in Y_ids into a onehot vector
    Y_seq_onehots = [to_categorical(seq, nb_classes=num_tags) for seq in Y_ids]

    # format X and Y data
    nb_samples = len(X_ids)
    X = create_data_matrix_X(X_ids        , nb_samples, maxlen, num_tags)
    Y = create_data_matrix_Y(Y_seq_onehots, nb_samples, maxlen, num_tags)

    # fit model
    print 'training begin'
    batch_size = 64
    history = lstm_model.fit(X, Y, batch_size=batch_size, nb_epoch=epochs, verbose=1)
    print 'training done'

    ######################################################################

    # temporary debugging-ness
    #lstm_model.load_weights('tmp_keras_weights')

    pred_classes = lstm_model.predict(X, batch_size=batch_size)

    predictions = []
    for i in range(nb_samples):
        num_words = len(Y_ids[i])
        tags = pred_classes[i,maxlen-num_words:].argmax(axis=1)
        #print 'gold: ', np.array(Y_ids[i])
        #print 'pred: ', tags
        #print
        predictions.append(tags.tolist())
    #print '\n\n\n\n'

    # confusion matrix
    confusion = np.zeros( (num_tags,num_tags) )
    for tags,yseq in zip(predictions,Y_ids):
        for y,p in zip(yseq, tags):
            confusion[p,y] += 1

    print ' '*6,
    for i in range(num_tags):
        print '%4d' % i,
    print ' (gold)'
    for i in range(num_tags):
        print '%2d' % i, '   ',
        for j in range(num_tags):
            print '%4d' % confusion[i][j],
        print
    print '(pred)'

    print '\n\n\n\n'

    precision = np.zeros(num_tags)
    recall    = np.zeros(num_tags)
    f1        = np.zeros(num_tags)

    for i in range(num_tags):
        correct    =     confusion[i,i]
        num_pred   = sum(confusion[i,:])
        num_actual = sum(confusion[:,i])

        p  = correct / (num_pred   + 1e-9)
        r  = correct / (num_actual + 1e-9)

        precision[i] = p
        recall[i]    = r
        f1[i]        = (2*p*r) / (p + r + 1e-9)

    print 'p: ',
    for p in precision: print '%4.2f' % p,
    print
    print 'r: ',
    for r in recall:    print '%4.2f' % r,
    print
    print 'f: ',
    for f in f1:        print '%4.2f' % f,
    print
    print 'avg-f1: ', np.mean(f1)
    print

    total_correct = sum( confusion[i,i] for i in range(num_tags) )
    total = confusion.sum()
    accuracy = total_correct / (total + 1e-9)
    print 'Accuracy: ', accuracy

    ######################################################################

    # needs to return something pickle-able
    lstm_model.save_weights('tmp_keras_weights')
    with open('tmp_keras_weights', 'rb') as f:
        lstm_model_str = f.read()
    os.remove('tmp_keras_weights')

    # information about fitting the model
    dev_score = {}
    dev_score['precision'] = precision
    dev_score['recall'   ] = recall
    dev_score['f1'       ] = f1
    dev_score['history'  ] = history.history

    # return model back to cliner
    keras_model_tuple = (lstm_model_str, input_dim, num_tags, maxlen)

    return keras_model_tuple, dev_score





def predict(keras_model_tuple, X_seq_ids):

    global lstm_model

    # unpack model metadata
    lstm_model_str, input_dim, num_tags, maxlen = keras_model_tuple

    # dump serialized model out to file in order to load it
    with open('lstm_keras_weights', 'wb') as f:
        f.write(lstm_model_str)

    # build LSTM once (weird errors if re-compiled many times)
    if lstm_model is None:
        lstm_model = create_bidirectional_lstm(input_dim, num_tags, maxlen)

    # load weights from serialized file
    lstm_model.load_weights('lstm_keras_weights')
    os.remove('lstm_keras_weights')

    # format data for LSTM
    nb_samples = len(X_seq_ids)
    X = create_data_matrix_X(X_seq_ids, nb_samples, maxlen, num_tags)

    # Predict tags using LSTM
    batch_size = 128
    #import cPickle as pickle
    #if os.path.exists(preds):
    #    with open(preds, 'rb') as f:
    #        p = pickle.load(f)
    #else:
    p = lstm_model.predict(X, batch_size=batch_size)
    #    with open(preds, 'wb') as f:
    #        pickle.dump(p, f)

    predictions = []
    for i in range(nb_samples):
        num_words = len(X_seq_ids[i])
        if num_words <= maxlen:
            tags = p[i,maxlen-num_words:].argmax(axis=1)
            predictions.append(tags.tolist())
        else:
            # if the sentence had more words than the longest sentence
            #   in the training set
            residual_zeros = [ 0 for _ in range(num_words-maxlen) ]
            padded = list(p[i].argmax(axis=1)) + residual_zeros
            predictions.append(padded)
    print predictions

    return predictions




def create_bidirectional_lstm(input_dim, nb_classes, maxlen, W=None):
    # model will expect: (nb_samples, timesteps, input_dim)

    # input tensor
    sequence = Input(shape=(maxlen,), dtype='int32')

    # initialize Embedding layer with pretrained vectors
    if W is not None:
        embedding_size = W.shape[1]
        weights = [W]
    else:
        embedding_size = 300
        weights = None

    # Embedding layer
    embedding = Embedding(output_dim=embedding_size, input_dim=input_dim, input_length=maxlen, mask_zero=True, weights=weights)(sequence)


    # LSTM 1 input
    hidden_units = 128
    lstm_f1 =LSTM(output_dim=hidden_units,return_sequences=True)(embedding)
    lstm_r1 =LSTM(output_dim=hidden_units,return_sequences=True,go_backwards=True)(embedding)
    merged1 = merge([lstm_f1, lstm_r1], mode='concat', concat_axis=-1)


    # LSTM 2 input
    lstm_f2 =LSTM(output_dim=hidden_units,return_sequences=True)(merged1)
    lstm_r2 =LSTM(output_dim=hidden_units,return_sequences=True,go_backwards=True)(merged1)
    merged2 = merge([lstm_f2, lstm_r2], mode='concat', concat_axis=-1)


    # Dropout
    after_dp = TimeDistributed(Dropout(0.5))(merged2)


    # fully connected layer
    fc1 = TimeDistributed(Dense(output_dim=128, activation='sigmoid'))(after_dp)
    fc2 = TimeDistributed(Dense(output_dim=nb_classes, activation='softmax'))(fc1)

    model = Model(input=sequence, output=fc2)

    print
    print 'compiling model'
    start = time.clock()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    #print '\tWARNING: skipping compilation'
    end = time.clock()
    print 'finished compiling: ', (end-start)
    print

    return model





def create_data_matrix_X(X_ids, nb_samples, maxlen, nb_classes):
    X = np.zeros((nb_samples, maxlen))

    for i in range(nb_samples):
        cur_len = len(X_ids[i])

        # ignore tail of sentences longer than what was trained on
        #    (only happens during prediction)
        if maxlen-cur_len < 0:
            cur_len = maxlen

        # We pad on the left with zeros,
        #    so for short sentences the first elemnts in the matrix are zeros
        X[i, maxlen - cur_len:] = X_ids[i][:maxlen]

    return X




def create_data_matrix_Y(Y_seq_onehots, nb_samples, maxlen, nb_classes):
    Y = np.zeros((nb_samples, maxlen, nb_classes))

    for i in range(nb_samples):
        cur_len = len(Y_seq_onehots[i])

        # We pad on the left with zeros,
        #    so for short sentences the first elemnts in the matrix are zeros
        Y[i, maxlen - cur_len:, :] = Y_seq_onehots[i]

    return Y



