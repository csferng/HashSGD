import collections
import math

BOOLEAN_FEATURE_ID = set([1,2,10,11,12,13,14,24,25,26,30,31,32,33,41,42,43,44,45,55,56,57,62,63,71,72,73,74,75,85,86,87,92,93,101,102,103,104,105,115,116,117,126,127,128,129,130,140,141,142])
CATEGORICAL_FEATURE_ID = set([3,4,34,35,61,64,65,91,94,95])
NUMERIC_FEATURE_ID = set([5,6,7,8,9,15,16,17,18,19,20,21,22,23,27,28,29,36,37,38,39,40,46,47,48,49,50,51,52,53,54,58,59,60,66,67,68,69,70,76,77,78,79,80,81,82,83,84,88,89,90,96,97,98,99,100,106,107,108,109,110,111,112,113,114,118,119,120,121,122,123,124,125,131,132,133,134,135,136,137,138,139,143,144,145])
INTEGER_FEATURE_ID = set([15,17,18,22,23,27,46,48,49,53,54,58,76,78,79,83,84,88,106,108,109,113,114,118,131,133,134,138,139,143])

FEATURE_TYPE = '-BBCCNNNNNBBBBBINIINNNIIBBBINNBBBBCCNNNNNBBBBBINIINNNIIBBBINNCBBCCNNNNNBBBBBINIINNNIIBBBINNCBBCCNNNNNBBBBBINIINNNIIBBBINNNNNNNBBBBBINIINNNIIBBBINN'

class FeatureTransformer(object):
    def __init__(self, d):
        self.dim = d + 1
        self.hash_base = d

    def transform(self, x, features):
        raise Exception("Not implemented transform() in base class")

    def hash_to_D(self, idx, feat):
        return abs(hash('%d_%s'%(idx,feat))) % self.hash_base

    def need_init(self): return False

    def init_per_train(self, feat_lines):
        if not self.need_init():
            return
        self._pre_init()
        for line in feat_lines:
            self._do_init(line)
        self._post_init()

    def _pre_init(self): pass
    def _do_init(self, line): pass
    def _post_init(self): pass

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
        # features[0] is ID
        # x[0] reserved for bias term
        if x is None:
            x = [(self.hash_base,1.)]*len(features)
        for (idx,feat) in enumerate(features):
            if idx > 0: # skip ID
                x[idx] = (self.hash_to_D(idx, feat), 1.)
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
        self.dim = self.hash_base + 146 # 1 bias, 145 features

    def transform(self, x, features):
        # features[0] is ID
        # x[0] reserved for bias term
        x = [(self.hash_base,1.)]
        for (idx,feat) in enumerate(features):
            if idx > 0: # skip ID
                x.append((self.hash_to_D(idx,feat), 1.))
            if feat == '' or feat == '0': continue
            if FEATURE_TYPE[idx] == 'B':
                x.append((self.hash_base+idx, 1. if feat=='YES' else -1.))
            elif FEATURE_TYPE[idx] == 'C':
                raw = int(feat)
                val = -1 if raw==-1 else math.log(raw+1)/self.scale[idx]
                x.append((self.hash_base+idx, val))
            elif FEATURE_TYPE[idx] == 'N':
                x.append((self.hash_base+idx, float(feat)/self.scale[idx]))
        return x

    def need_init(self): return True

    def _pre_init(self):
        self.minmax = collections.defaultdict(lambda:(1e99,-1e99))

    def _do_init(self, line):
        features = line.rstrip().split(',')    # features[0] is ID
        for i in NUMERIC_FEATURE_ID:    # feature id here is 1-based indexed
            val = features[i]
            if val == '': continue
            x = float(val)
            mm = self.minmax[i]
            self.minmax[i] = (min(x,mm[0]), max(x,mm[1]))

    def _post_init(self):
        for i in NUMERIC_FEATURE_ID:
            if i in INTEGER_FEATURE_ID:
                self.scale[i] = math.log(self.minmax[i][1]+1)
            else:
                self.scale[i] = max(abs(self.minmax[i][0]), abs(self.minmax[i][1]))
        del self.minmax

def get_maker(D, transform_method):
    if transform_method == 'onehot':
        return OneHotTransformer(D)
    elif transform_method == 'numeric':
        return NumericValueTransformer(D)
    else:
        raise ValueError("Unknown transform method: "+transform_method)
