import sys

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
        x[m+1] = abs(hash(str(m+1) + '_' + feat)) % D
    return x

transform = one_hot
D = 262147  # prime near 2**18

def init(_D, transform_method):
    global D, transform
    D = _D
    if transform_method == 'one_hot':
        transform = one_hot
