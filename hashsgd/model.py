""" XXX fix import error """
import _hashlib
_hashlib.openssl_md_meth_names = frozenset(['SHA256', 'SHA512', 'sha256', 'sha512', 'md5', 'SHA1', 'SHA224', 'SHA', 'SHA384', 'sha1', 'sha224', 'sha', 'MD5', 'sha384'])
""" XXX """
import itertools
from math import exp, log, sqrt
import random

class LogisticRegressionModel(object):
    def __init__(self, labels, D):
        self.labels = labels
        self.D = D
        self.W = [ [0.]*D for _ in labels ] # weights
        self.N = [ [0.]*D for _ in labels ] # sum of abs(gradient) for a given feature, used for adaptive learning rate
        self.__space = [0]*len(labels)

    def _predict_one(self, x, w):
        wTx = sum( w[i]*v for (i,v) in x )  # wTx = w[i]*x[i]
        return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid

    def predict(self, x):
        """ Get probability estimation on x.
        @param x: features, in sparse format [(idx,value)*]
        @return: probability of p(y=1 | x; w)
        """
        for (i,w) in enumerate(self.W):
            self.__space[i] = self._predict_one(x, w)
        return self.__space

    def _update_one(self, alpha, w, n, x, p, y):
        for (i,v) in x:
            # alpha / sqrt(n) is the adaptive learning rate
            # (p - y) * x[i] is the current gradient
            n[i] += ((p-y)*v)**2
            w[i] -= (p - y) * v * alpha / sqrt(n[i])

    def update(self, x, y, alpha):
        """ Update the model.
        @param x: features, in sparse format [(idx,value)*]
        @param y: truth labels
        @param alpha: learning rate
        """
        losses = self.predict(x)
        for (i,(p,k)) in enumerate(itertools.izip(losses,self.labels)):
            self._update_one(alpha, self.W[i], self.N[i], x, p, y[k])
            p = max(min(p, 1.-1e-15), 1e-15)    # bounded
            losses[i]  = -log(p if y[k]==1. else (1.-p))
        return losses

    def loss(self, x, y):
        """ Calculate log-loss.
        @param x: features, in sparse format [(idx,value)*]
        @param y: truth labels
        """
        losses = self.predict(x)
        for (i,(p,k)) in enumerate(itertools.izip(losses,self.labels)):
            p = max(min(p, 1.-1e-15), 1e-15)    # bounded
            losses[i]  = -log(p if y[k]==1. else (1.-p))
        return losses

class FactorizationMachineModel(LogisticRegressionModel):
    def __init__(self, labels, D, K=4):
        super(FactorizationMachineModel, self).__init__(labels, D)
        self.K = K
        del self.W
        del self.N
        # W[l][k][d], for a feature dim d and a label l, the weight vector is w[l][*][d]
        self.W = [ [ [ random.gauss(0., .01) for d in xrange(D) ] for k in xrange(K) ] for l in labels ]
        self.N = [ [ [0]*D for k in xrange(K) ] for l in labels ]
        self.__space_wtx = [0.]*K

    def _predict_one(self, x, w):
        """ Calculate sum_{i!=j} w_i^Tw_jx_ix_j """
        wTx = self.__space_wtx
        for (k,w_k) in enumerate(w):
            wTx[k] = sum( w_k[i]*v for (i,v) in x )     # wTx[k] = \sum_i w[k][i]*x[i]
        wTx_2 = sum( (w_k[i]*v)**2 for w_k in w for (i,v) in x )  # wTx_2 = \sum_i \sum_k (w[k][i]*x[i])^2

        wTwxx = (sum( wx_k**2 for wx_k in wTx ) - wTx_2) / 2.
        return 1. / (1. + exp(-max(min(wTwxx, 20.), -20.)))  # bounded sigmoid

    def _update_one(self, alpha, w, n, x, p, y):
        wTx = self.__space_wtx
        for (k,w_k) in enumerate(w):
            wTx[k] = sum( w_k[i]*v for (i,v) in x )     # wTx[k] = \sum_i w[k][i]*x[i]
        diff = p - y
        # grad of w[*][i] = (y-p)(\sum_{j!=i} w[*][j]x[j] )
        for (w_k, n_k, wTx_k) in itertools.izip(w, n, wTx):
            for (i,v) in x:
                grad = diff * (wTx_k-w_k[i]*v)
                # alpha / sqrt(n) is the adaptive learning rate
                n_k[i] += (grad)**2
                w_k[i] -= 0 if grad==0 else grad * alpha / (n_k[i]**(1./4))
