from math import exp, log, sqrt

class LogisticRegressionModel(object):
    def __init__(self, labels, D):
        self.labels = labels
        self.D = D
        self.W = [ [0.]*D for _ in labels ] # weights
        self.N = [ [0.]*D for _ in labels ] # sum of abs(gradient) for a given feature, used for adaptive learning rate

    def _predict_one(self, x, w):
        wTx = sum( w[i]*v for (i,v) in x )  # wTx = w[i]*x[i]
        return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid

    def predict(self, x):
        """ Get probability estimation on x.
        @param x: features, in sparse format [(idx,value)*]
        @return: probability of p(y=1 | x; w)
        """
        return [ self._predict_one(x, w) for w in self.W ]

    def _update_one(self, alpha, w, n, x, p, y):
        for (i,v) in x:
            # alpha / sqrt(n) is the adaptive learning rate
            # (p - y) * x[i] is the current gradient
            n[i] += abs((p-y)*v)
            w[i] -= (p - y) * v * alpha / sqrt(n[i])

    def update(self, x, y, alpha):
        """ Update the model.
        @param x: features, in sparse format [(idx,value)*]
        @param y: truth labels
        @param alpha: learning rate
        """
        l = []
        for (k,w,n) in zip(self.labels, self.W, self.N):
            p = self._predict_one(x, w)
            self._update_one(alpha, w, n, x, p, y[k])
            p = max(min(p, 1. - 10e-15), 10e-15)    # bounded
            l.append(-log(p) if y[k]==1. else -log(1.-p))
        return l

    def loss(self, x, y):
        """ Calculate log-loss.
        @param x: features, in sparse format [(idx,value)*]
        @param y: truth labels
        """
        l = []
        for (k,w,n) in zip(self.labels, self.W, self.N):
            p = self._predict_one(x, w)
            p = max(min(p, 1. - 10e-15), 10e-15)    # bounded
            l.append(-log(p) if y[k]==1. else -log(1.-p))
        return l
