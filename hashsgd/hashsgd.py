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

from data import PoolStreamData as TrainData, StreamData as TestData
from model import LogisticRegressionModel as Model
import feature_transformer
import util

from datetime import datetime
from math import log, exp, sqrt

import argparse

# TL; DR
# the main learning process start at line 122


# parameters #################################################################

# a list for range(0, 33) - 13, no need to learn y14 since it is always 0
K = [k for k in range(33) if k != 13]

alpha = .1   # learning rate for sgd optimization

feature_maker = None

def info(s):
    print('%s\t%s' % (datetime.now().strftime('%m/%d %H:%M:%S'), s))

def parse_args():
    global alpha

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', help='number of weights used for each model, we have 32 of them', type=int, default=262147)
    parser.add_argument('-R', help='number of epoches for SGD', type=int, default=2)
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

# training and testing #######################################################
def train_one(data, model):
    loss = 0.
    loss_y14 = log(1. - 10**-15)

    cnt = 0
    for (i,(ID, x, y)) in enumerate(data):
        cnt += 1

        loss += sum(model.update(x, y, alpha))   # report current loss and update model
        loss += loss_y14  # the loss of y14, logloss is never zero

        # print out progress, so that we know everything is working
        if cnt % 100000 == 0:
            info('trained now: %d\tcurrent logloss: %f'%(i+1, loss/33./cnt))
            loss = 0.
            cnt = 0

    info('trained all: %d\tcurrent logloss: %f'%(cnt, loss/33./max(1,cnt)))
    return model

def evaluate(valid_data, model):
    loss = 0.
    loss_y14 = log(1. - 10**-15)

    cnt = 0
    for (ID, x, y) in valid_data:
        cnt += 1

        loss += sum(model.loss(x,y))    # sum loss of all labels except y14
        loss += loss_y14  # the loss of y14, logloss is never zero

    info('evaluated: %d\tlogloss: %f'%(cnt, loss/33./cnt))
    return (cnt, loss/33.)

def new_model(D):
    return Model(K, D)

def main():
    global feature_maker
    info('start')
    start = datetime.now()

    args = parse_args()
    feature_maker = feature_transformer.get_maker(args.D, args.transform)

    if args.cmd == 'test':   # train on training data and predict on testing data
        feature_maker.init_per_train(util.open_csv(args.train))
        data = TrainData(args.train, feature_maker, args.train_label)
        model = new_model(feature_maker.dim)
        for r in xrange(args.R):
            if r > 0: data.rewind()
            model = train_one(data, model)
        with open(args.prediction, 'w') as outfile:
            outfile.write('id_label,pred\n')
            for ID, x in TestData(args.test, feature_maker):
                pred = model.predict(x)
                for (k,p) in zip(K,pred):
                    outfile.write('%s_y%d,%.16f\n' % (ID,k+1,p))
                    if k == 12:
                        outfile.write('%s_y14,0.0\n' % ID)
    else:   # do cross validation
        nfold = args.nfold
        cnt_ins = [0]*args.R
        cnt_loss = [0.]*args.R
        for fold in xrange(1,nfold+1):
            feature_maker.init_per_train(util.open_csv(args.train, (-fold,nfold)))
            train_data = TrainData(args.train, feature_maker, args.train_label, (-fold,nfold))
            model = new_model(feature_maker.dim)
            valid_data = TestData(args.train, feature_maker, args.train_label, (fold,nfold))
            for r in xrange(args.R):
                if r > 0: train_data.rewind()
                model = train_one(train_data, model)
                if r > 0: valid_data.rewind()
                f_ins, f_loss = evaluate(valid_data, model)
                info("round validation: %f" % (f_loss/f_ins))
                cnt_ins[r] += f_ins
                cnt_loss[r] += f_loss
            del train_data
            del valid_data
            del model
        for r in xrange(args.R):
            print "%d round CV result: %f"%(r, cnt_loss[r]/cnt_ins[r])

    info('Done, elapsed time: %s' % str(datetime.now() - start))

if __name__== '__main__':
    import traceback
    try:
        main()
#        import cProfile
#        cProfile.run('main()')
    except:
        traceback.print_exc()
