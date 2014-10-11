import itertools

def in_fold(i, fold_in_cv):
    x, nf = fold_in_cv
    r = (i%nf) + 1  # make r in 1 ~ nf instead of 0 ~ nf-1
    if x > 0:   # validation, return only x-th fold
        return r == x
    elif x < 0: # training, return everything other than x-th fold
        return r != -x
    else:
        raise ValueError("Wrong fold setting: "+repr(fold_in_cv))

def open_csv(path, fold_in_cv=None):
    with open(path) as f:
        f.readline()    # skip header
        for (i,line) in enumerate(f):
            if fold_in_cv is None or in_fold(i, fold_in_cv):
                yield line

def open_feature_and_label(feat_path, label_path, fold_in_cv=None):
    feats = open_csv(feat_path, fold_in_cv)
    labels = open_csv(label_path, fold_in_cv)
    for pair in itertools.izip(feats, labels):
        yield pair
