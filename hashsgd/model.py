from math import exp, log, sqrt

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
            n[i] += abs((p-y)*v)
            w[i] -= (p - y) * v * alpha / sqrt(n[i])

    def update(self, x, y, alpha):
        """ Update the model.
        @param x: features, in sparse format [(idx,value)*]
        @param y: truth labels
        @param alpha: learning rate
        """
        losses = self.predict(x)
        for (i,(p,k)) in enumerate(zip(losses,self.labels)):
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
        for (i,(p,k)) in enumerate(zip(losses,self.labels)):
            p = max(min(p, 1.-1e-15), 1e-15)    # bounded
            losses[i]  = -log(p if y[k]==1. else (1.-p))
        return losses
