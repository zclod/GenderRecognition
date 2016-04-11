from sklearn import svm
from siamese_model import build_model
import json

def load_pkl_video(dir):
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)

    file_load = open(dir + "/prova.pkl", 'rb')
    m = pickle.load(file_load)
    return m

with open("directories.json", "r") as json_file:
    dirs = json.load(json_file)

m = build_model((3,128,48))

x, y = load_pkl_video(dirs["train"][0])

predictions = m.predict(x)
print(predictions)
