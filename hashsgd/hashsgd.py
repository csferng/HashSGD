'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''

import feature_transformer

from datetime import datetime
from math import log, exp, sqrt

import argparse

# TL; DR
# the main learning process start at line 122


# parameters #################################################################

train = 'train.csv'  # path to training file
label = 'trainLabels.csv'  # path to label file of training data
test = 'test.csv'  # path to testing file
prediction = 'prediction.csv'   # path to prediction file

D = 2 ** 18  # number of weights use for each model, we have 32 of them
alpha = .1   # learning rate for sgd optimization

def parse_args():
    global train, label, test, prediction, D, alpha

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', help='number of weights used for each model, we have 32 of them', type=int, default=262147)
    parser.add_argument('--alpha', '-a', help='learning rate for SGD optimization', type=float, default=.1)
    parser.add_argument('--transform', help='method to transform features', default='one_hot')
    parser.add_argument('train', help='path to training file')
    parser.add_argument('train_label', help='path to label file of training data')
    subparsers = parser.add_subparsers()
    subparser_cv = subparsers.add_parser('cv', help='do cross validation')
    subparser_cv.add_argument('--nfold', '-v', help='number of folds', type=int, default=5)
    subparser_cv.set_defaults(cmd='cv')
    subparser_test = subparsers.add_parser('test', help='do prediction on test data')
    subparser_test.add_argument('test', help='path to label testing file')
    subparser_test.add_argument('prediction', help='path to prediction file')
    subparser_test.set_defaults(cmd='test')
    args = parser.parse_args()
    print args
    train = args.train
    label = args.train_label
    D = args.D
    alpha = args.alpha
    if args.cmd == 'test':
        test = args.test
        prediction = args.prediction
    return args


# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, label_path=None):
    for t, line in enumerate(open(path)):
        # initialize our generator
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
            x = [0] * 146
            if label_path:
                label = open(label_path)
                label.readline()  # we don't need the headers
            continue
        # parse x
        features = line.rstrip().split(',')
        ID = int(features[0])
        x = feature_transformer.transform(x, features[1:])
        # parse y, if provided
        if label_path:
            # use float() to prevent future type casting, [1:] to ignore id
            y = [float(y) for y in label.readline().split(',')[1:]]
        yield (ID, x, y) if label_path else (ID, x)


# B. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     bounded logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def predict(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
# alpha: learning rate
#     w: weights
#     n: sum of previous absolute gradients for a given feature
#        this is used for adaptive learning rate
#     x: feature, a list of indices
#     p: prediction of our model
#     y: answer
# MODIFIES:
#     w: weights
#     n: sum of past absolute gradients
def update(alpha, w, n, x, p, y):
    for i in x:
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
        n[i] += abs(p - y)
        w[i] -= (p - y) * 1. * alpha / sqrt(n[i])


# training and testing #######################################################
def train_one(train, label, fold_in_cv=None):
    # a list for range(0, 33) - 13, no need to learn y14 since it is always 0
    K = [k for k in range(33) if k != 13]

    # initialize our model, all 32 of them, again ignoring y14
    w = [[0.] * D if k != 13 else None for k in range(33)]
    n = [[0.] * D if k != 13 else None for k in range(33)]

    loss = 0.
    loss_y14 = log(1. - 10**-15)

    cnt = 0
    for (i,(ID, x, y)) in enumerate(data(train, label)):
        # skip validation data in training
        if fold_in_cv is not None and i%fold_in_cv[1]==fold_in_cv[0]:
            continue
        cnt += 1

        # get predictions and train on all labels
        for k in K:
            p = predict(x, w[k])
            update(alpha, w[k], n[k], x, p, y[k])
            loss += logloss(p, y[k])  # for progressive validation
        loss += loss_y14  # the loss of y14, logloss is never zero

        # print out progress, so that we know everything is working
        if cnt % 100000 == 0:
            print('%s\ttrained now: %d\tcurrent logloss: %f' % (
                datetime.now(), cnt, (loss/33.)/cnt))
    print('%s\ttrained all: %d\tcurrent logloss: %f' % (
        datetime.now(), cnt, (loss/33.)/cnt))

    return w

def evaluate(valid_data, label, w, fold_in_cv=None):
    # a list for range(0, 33) - 13, no need to learn y14 since it is always 0
    K = [k for k in range(33) if k != 13]

    loss = 0.
    loss_y14 = log(1. - 10**-15)

    cnt = 0
    for (i,(ID, x, y)) in enumerate(data(valid_data, label)):
        # skip training data
        if fold_in_cv is not None and i%fold_in_cv[1]!=fold_in_cv[0]:
            continue
        cnt += 1

        # get predictions and train on all labels
        for k in K:
            p = predict(x, w[k])
            loss += logloss(p, y[k])  # for progressive validation
        loss += loss_y14  # the loss of y14, logloss is never zero

    print('%s\tevaluated: %d\tcurrent logloss: %f' % (
        datetime.now(), cnt, (loss/33.)/cnt))
    return (cnt, loss/33.)

def main():
    start = datetime.now()
    print('%s\tstart' % (start))

    args = parse_args()
    feature_transformer.init(args.D, args.transform)

    if args.cmd == 'test':   # train on training data and predict on testing data
        w = train_one(args.train, args.train_label)
        with open(args.prediction, 'w') as outfile:
            outfile.write('id_label,pred\n')
            for ID, x in data(args.test):
                for k in K:
                    p = predict(x, w[k])
                    outfile.write('%s_y%d,%s\n' % (ID, k+1, str(p)))
                    if k == 12:
                        outfile.write('%s_y14,0.0\n' % ID)
    else:   # do cross validation
        nfold = args.nfold
        cnt_ins = 0
        cnt_loss = 0.
        for fold in xrange(nfold):
            w = train_one(args.train, args.train_label, (fold,nfold))
            f_ins, f_loss = evaluate(args.train, args.train_label, w, (fold,nfold))
            cnt_ins += f_ins
            cnt_loss += f_loss
        print "CV result: %f"%(cnt_loss/cnt_ins)

    print('Done, elapsed time: %s' % str(datetime.now() - start))

if __name__== '__main__':
    main()
