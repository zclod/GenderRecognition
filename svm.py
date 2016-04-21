from sklearn import svm
from siamese_model import build_model
import json
import numpy as np
import random

def savepkl(obj, path_out):
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)

    file_save = open(path_out, 'wb')
    pickle.dump(obj, file_save, -1)

    file_save.close()


def load_pkl_video(dir):
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)

    file_load = open(dir + "/prova.pkl", 'rb')
    m = pickle.load(file_load)
    return m

def load_pkl(file):
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)

    file_load = open(file, 'rb')
    m = pickle.load(file_load)
    return m

try:
    samples, labels = load_pkl("svm_train.pkl")
except:
    samples = []

if not len(samples):
    m = build_model((3,128,48), "my_model_weights_3.h5")
    labels = np.empty((0,1))
    samples = np.empty((0,32))

    with open("directories.json", "r") as json_file:
        dirs = json.load(json_file)
    for d in random.sample(dirs["train"],len(dirs["train"])):
        (x, y) = load_pkl_video(d)
        if len(y) > 0:
            indexes = random.sample(range(0,len(y)), min(500, len(y)))
            predictions = m.predict(x)
            for i in indexes:
                labels = np.append(labels, y[i])
                samples = np.vstack([samples, predictions[i]])

    savepkl((samples, labels), "svm_train.pkl")

x_tr = samples[0:5000,:]
y_tr = labels[0:5000]
x_ts = samples[5000:,:]
y_ts = labels[5000:]

classificatore = svm.SVC()
classificatore.fit(x_tr, y_tr)

print("fine train")
