import util

import itertools

class Data(object):
    """ Container of a data set. """
    def __init__(self, feat_path, feature_maker, label_path=None, fold_in_cv=None):
        """ Construct a dat set.
        parameters:
        feat_path: path to raw data file
        feature_maker: instance of FeatureTransformer to convert raw data to features
        label_path: (optional) path to label file
        fold_in_cv: (optional) tuple (x,y). Split data to y folds, and take
            only x-th fold if x>0, or all except x-th fold if x<0
        """
        self.instances = []
        feat_lines = util.open_csv(feat_path, fold_in_cv)
        if label_path:
            label_lines = util.open_csv(label_path, fold_in_cv)
        else:
            label_lines = itertools.repeat(None)   # endless None
        for (feat_line, label_line) in itertools.izip(feat_lines, label_lines):
            # parse x
            features = feat_line.rstrip().split(',')
            ID = int(features[0])
            x = feature_maker.transform(None, features[1:])
            if label_line is None:
                self.instances.append((ID, x))
            else:
                # parse y, if provided
                # use float() to prevent future type casting, [1:] to ignore id
                y = [float(y) for y in label_line.split(',')[1:]]
                self.instances.append((ID, x, y))

    def __iter__(self):
        """ Iterator of the data set. (ID,x,y) or (ID,x). """
        return itertools.chain(self.instances)
