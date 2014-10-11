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
import util

from datetime import datetime
from math import log, exp, sqrt

import argparse

# TL; DR
# the main learning process start at line 122


# parameters #################################################################

alpha = .1   # learning rate for sgd optimization

feature_maker = None

def info(s):
    print('%s\t%s' % (datetime.now().strftime('%m/%d %H:%M:%S'), s))

def parse_args():
    global train, label, test, prediction, alpha

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
    alpha = args.alpha
    print args
    return args


# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
#     fold_in_cv: (optional) tuple (x,y), see util.in_fold
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, label_path=None, fold_in_cv=None):
    # create a static x,
    # so we don't have to construct a new x for every instance
    # each item of x should be a tuple (feat-id, feat-value)
    x = [0] * 146
    x[0] = (0,1.)
    if label_path:
        line_stream = util.open_feature_and_label(path, label_path, fold_in_cv)
    else:
        line_stream = util.open_csv(path)
    for line in line_stream:
        feat_line = line[0] if label_path else line
        # parse x
        features = feat_line.rstrip().split(',')
        ID = int(features[0])
        x = feature_maker.transform(x, features[1:])
        if label_path is None:
            yield (ID, x)
        else:
            # parse y, if provided
            # use float() to prevent future type casting, [1:] to ignore id
            y = [float(y) for y in line[1].split(',')[1:]]
            yield (ID, x, y)

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
    for (i,v) in x:  # do wTx
        wTx += w[i] * v  # w[i] * x[i]
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
    for (i,v) in x:
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        n[i] += abs(p - y)
        w[i] -= (p - y) * v * alpha / sqrt(n[i])


# training and testing #######################################################
def train_one(train, label, fold_in_cv=None):
    # a list for range(0, 33) - 13, no need to learn y14 since it is always 0
    K = [k for k in range(33) if k != 13]

    # initialize our model, all 32 of them, again ignoring y14
    D = feature_maker.dim
    w = [[0.] * D if k != 13 else None for k in range(33)]
    n = [[0.] * D if k != 13 else None for k in range(33)]

    loss = 0.
    loss_y14 = log(1. - 10**-15)

    cnt = 0
    for (i,(ID, x, y)) in enumerate(data(train, label, fold_in_cv)):
        cnt += 1

        # get predictions and train on all labels
        for k in K:
            p = predict(x, w[k])
            update(alpha, w[k], n[k], x, p, y[k])
            loss += logloss(p, y[k])  # for progressive validation
        loss += loss_y14  # the loss of y14, logloss is never zero

        # print out progress, so that we know everything is working
        if cnt % 100000 == 0:
            info('trained now: %d\tcurrent logloss: %f'%(cnt, loss/33./cnt))

    info('trained all: %d\tcurrent logloss: %f'%(cnt, loss/33./cnt))
    return w

def evaluate(valid_data, label, w, fold_in_cv=None):
    # a list for range(0, 33) - 13, no need to learn y14 since it is always 0
    K = [k for k in range(33) if k != 13]

    loss = 0.
    loss_y14 = log(1. - 10**-15)

    cnt = 0
    for (ID, x, y) in data(valid_data, label, fold_in_cv):
        cnt += 1

        # get predictions and train on all labels
        for k in K:
            p = predict(x, w[k])
            loss += logloss(p, y[k])  # for progressive validation
        loss += loss_y14  # the loss of y14, logloss is never zero

    info('evaluated: %d\tlogloss: %f'%(cnt, loss/33./cnt))
    return (cnt, loss/33.)

def main():
    global feature_maker
    info('start')
    start = datetime.now()

    args = parse_args()
    feature_maker = feature_transformer.get_maker(args.D, args.transform)

    if args.cmd == 'test':   # train on training data and predict on testing data
        feature_maker.initialize_per_train(util.open_csv(args.train))
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
        for fold in xrange(1,nfold+1):
            feature_maker.initialize_per_train(util.open_csv(args.train, (-fold,nfold)))
            w = train_one(args.train, args.train_label, (-fold,nfold))
            f_ins, f_loss = evaluate(args.train, args.train_label, w, (fold,nfold))
            cnt_ins += f_ins
            cnt_loss += f_loss
        print "CV result: %f"%(cnt_loss/cnt_ins)

    info('Done, elapsed time: %s' % str(datetime.now() - start))

if __name__== '__main__':
    main()
