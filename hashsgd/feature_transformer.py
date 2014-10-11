import collections
import math
import sys

BOOLEAN_FEATURE_ID = set([1,2,10,11,12,13,14,24,25,26,30,31,32,33,41,42,43,44,45,55,56,57,62,63,71,72,73,74,75,85,86,87,92,93,101,102,103,104,105,115,116,117,126,127,128,129,130,140,141,142])
CATEGORICAL_FEATURE_ID = set([3,4,34,35,61,64,65,91,94,95])
NUMERIC_FEATURE_ID = set([5,6,7,8,9,15,16,17,18,19,20,21,22,23,27,28,29,36,37,38,39,40,46,47,48,49,50,51,52,53,54,58,59,60,66,67,68,69,70,76,77,78,79,80,81,82,83,84,88,89,90,96,97,98,99,100,106,107,108,109,110,111,112,113,114,118,119,120,121,122,123,124,125,131,132,133,134,135,136,137,138,139,143,144,145])
INTEGER_FEATURE_ID = set([15,17,18,22,23,27,46,48,49,53,54,58,76,78,79,83,84,88,106,108,109,113,114,118,131,133,134,138,139,143])

NUMERIC_SCALE = collections.defaultdict(lambda:1)

def set_scale(feat_lines):
    global NUMERIC_SCALE
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
            NUMERIC_SCALE[i] = math.log(minmax[i][1]+1)
        else:
            NUMERIC_SCALE[i] = max(abs(minmax[i][0]), abs(minmax[i][1]))

def hash_to_D(idx, feat):
    return abs(hash(str(idx) + '_' + feat)) % D

def one_hot(x, features):
    """ one-hot encode everything with hash trick
    categorical: one-hotted
    boolean: ONE-HOTTED
    numerical: ONE-HOTTED!
    note, the build in hash(), although fast is not stable,
          i.e., same value won't always have the same hash
          on different machines
    """
    # x[0] reserved for bias term
    for (m,feat) in enumerate(features):
        idx = m + 1
        featid = hash_to_D(idx, feat)
        x[idx] = (featid, 1.)
    return x

def hash_and_value(x, features):
    """
    categorical: one-hot encoded
    boolean: one-hot indicator for missing, or +1/-1 for YES/NO
    numerical: one-hot indicator for missing, or scaled to -1~+1 with 0 unchanged
    """
    # x[0] reserved for bias term
    for (m,feat) in enumerate(features):
        idx = m + 1
        if idx in CATEGORICAL_FEATURE_ID:
            featid = hash_to_D(idx, feat)
            featval = 1.
        elif idx in BOOLEAN_FEATURE_ID:
            featid = hash_to_D(idx, '' if feat=='' else 'YES')
            featval = 1. if feat in ['','YES'] else -1.
        elif idx in INTEGER_FEATURE_ID:
            featid = hash_to_D(idx, '' if feat in ['','-1'] else '0')
            featval = 1. if feat=='' or feat=='-1' else math.log(int(feat)+1)/NUMERIC_SCALE[idx]
        elif idx in NUMERIC_FEATURE_ID:
            featid = hash_to_D(idx, '' if feat=='' else '0')
            featval = 1. if feat=='' else float(feat)/NUMERIC_SCALE[idx]
        else:
            print >> sys.stderr, "Not handled index:", idx
            sys.exit(1)
        x[idx] = (featid, featval)
    return x

transform = one_hot
D = 262147  # prime near 2**18

def init(_D, transform_method):
    global D, transform
    D = _D
    if transform_method == 'one_hot':
        transform = one_hot
    elif transform_method == 'hash_and_value':
        transform = hash_and_value
    else:
        print >> sys.stderr, "Unknown tranform method:", transform_method
        sys.exit(1)
