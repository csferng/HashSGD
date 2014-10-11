import util

import itertools

class Data(object):
    """ Container of a data set. """
    def __init__(self, feat_path, feature_maker, label_path=None, fold_in_cv=None):
        """ Construct a dat set.
        @param feat_path: path to raw data file
        @param feature_maker: instance of FeatureTransformer to convert raw data to features
        @param label_path: (optional) path to label file
        @param fold_in_cv: (optional) tuple (x,y). Split data to y folds, and take
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
            x = feature_maker.transform(None, features)
            if label_line is None:
                self.instances.append((ID, x))
            else:
                # parse y, if provided
                # use float() to prevent future type casting, [1:] to ignore id
                y = map(float, label_line.split(',')[1:])
                self.instances.append((ID, x, y))

    def __iter__(self):
        """ Return an iterator of the data set. (ID,x,y) or (ID,x). """
        return iter(self.instances)

class StreamData(object):
    """ Container of a streaming data set. """
    def __init__(self, feat_path, feature_maker, label_path=None, fold_in_cv=None):
        """ Construct a dat set.
        @param feat_path: path to raw data file
        @param feature_maker: instance of FeatureTransformer to convert raw data to features
        @param label_path: (optional) path to label file
        @param fold_in_cv: (optional) tuple (x,y). Split data to y folds, and take
            only x-th fold if x>0, or all except x-th fold if x<0
        """
        self.feat_path = feat_path
        self.feature_maker = feature_maker
        self.label_path = label_path
        self.fold_in_cv = fold_in_cv
        self.feat_lines = None
        self.label_lines = None

        self.initialize()

    def __iter__(self):
        """ Return an iterator of the data set. (ID,x,y) or (ID,x). """
        return self

    def initialize(self):
        self.feat_lines = util.open_csv(self.feat_path, self.fold_in_cv)
        if self.label_path:
            self.label_lines = util.open_csv(self.label_path, self.fold_in_cv)
        else:
            self.label_lines = itertools.repeat(None)   # endless None

    def next(self):
        feat_line = self.feat_lines.next()
        label_line = self.label_lines.next()
        features = feat_line.rstrip().split(',')
        ID = int(features[0])
        x = self.feature_maker.transform(None, features)
        if label_line is None:
            return (ID, x)
        else:
            # parse y, if provided
            # use float() to prevent future type casting, [1:] to ignore id
            y = map(float, label_line.split(',')[1:])
            return (ID, x, y)
