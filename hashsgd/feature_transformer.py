import collections
import math

BOOLEAN_FEATURE_ID = set([1,2,10,11,12,13,14,24,25,26,30,31,32,33,41,42,43,44,45,55,56,57,62,63,71,72,73,74,75,85,86,87,92,93,101,102,103,104,105,115,116,117,126,127,128,129,130,140,141,142])
CATEGORICAL_FEATURE_ID = set([3,4,34,35,61,64,65,91,94,95])
NUMERIC_FEATURE_ID = set([5,6,7,8,9,15,16,17,18,19,20,21,22,23,27,28,29,36,37,38,39,40,46,47,48,49,50,51,52,53,54,58,59,60,66,67,68,69,70,76,77,78,79,80,81,82,83,84,88,89,90,96,97,98,99,100,106,107,108,109,110,111,112,113,114,118,119,120,121,122,123,124,125,131,132,133,134,135,136,137,138,139,143,144,145])
INTEGER_FEATURE_ID = set([15,17,18,22,23,27,46,48,49,53,54,58,76,78,79,83,84,88,106,108,109,113,114,118,131,133,134,138,139,143])

class FeatureTransformer(object):
    def __init__(self, d):
        self.dim = d
        self.hash_base = d

    def transform(self, x, features):
        raise Exception("Not implemented transform() in base class")

    def hash_to_D(self, idx, feat):
        return abs(hash(str(idx) + '_' + feat)) % self.hash_base

    def initialize_per_train(self, feat_lines):
        pass

class OneHotTransformer(FeatureTransformer):
    """ one-hot encode everything with hash trick
    categorical: one-hotted
    boolean: ONE-HOTTED
    numerical: ONE-HOTTED!
    note, the build in hash(), although fast is not stable,
          i.e., same value won't always have the same hash
          on different machines
    """
    def __init__(self, d):
        super(OneHotTransformer, self).__init__(d)

    def transform(self, x, features):
        # x[0] reserved for bias term
        if x is None:
            x = [(0,1.)] + [0]*len(features)
        for (m,feat) in enumerate(features):
            idx = m + 1
            featid = self.hash_to_D(idx, feat)
            x[idx] = (featid, 1.)
        return x

class NumericValueTransformer(FeatureTransformer):
    """
    categorical: one-hot encoded
    boolean: one-hot encoded, and +1/-1 for YES/NO
    integer: one-hot encoded, and log-scale to 0~1 with -1 unchanged
    numerical: one-hot encoded, and scaled to -1~+1 with 0 unchanged
    """
    def __init__(self, d):
        super(NumericValueTransformer, self).__init__(d)
        self.scale = collections.defaultdict(lambda:1)
        self.dim = self.hash_base + 145

    def transform(self, x, features):
        # x[0] reserved for bias term
        x = [(0,1.)]
        for (m,feat) in enumerate(features):
            idx = m + 1
            x.append((self.hash_to_D(idx,feat), 1.))
            if feat == '':
                pass
            elif idx in BOOLEAN_FEATURE_ID:
                x.append((self.hash_base+idx-1, 1. if feat=='YES' else -1.))
            elif idx in INTEGER_FEATURE_ID:
                raw = int(feat)
                val = -1 if raw==-1 else math.log(raw+1)/self.scale[idx]
                x.append((self.hash_base+idx-1, val))
            elif idx in NUMERIC_FEATURE_ID:
                x.append((self.hash_base+idx-1, float(feat)/self.scale[idx]))
        return x

    def initialize_per_train(self, feat_lines):
        minmax = collections.defaultdict(lambda:(1e99,-1e99))
        for line in feat_lines:
            features = line.rstrip().split(',')    # features[0] is ID
            for i in NUMERIC_FEATURE_ID:    # feature id here is 1-based indexed
                val = features[i]
                if val == '': continue
                x = float(val)
                mm = minmax[i]
                minmax[i] = (min(x,mm[0]), max(x,mm[1]))
        for i in NUMERIC_FEATURE_ID:
            if i in INTEGER_FEATURE_ID:
                self.scale[i] = math.log(minmax[i][1]+1)
            else:
                self.scale[i] = max(abs(minmax[i][0]), abs(minmax[i][1]))

def get_maker(D, transform_method):
    if transform_method == 'one_hot':
        return OneHotTransformer(D)
    elif transform_method == 'numeric_value':
        return NumericValueTransformer(D)
    else:
        raise ValueError("Unknown transform method: "+transform_method)
