import VIPerDS
import os
import cPickle as pickle
import sys
import numpy as np

video_directory = "/home/cla/Downloads/CMD/CMD/CMD"
dataset = np.empty([0, 3, 128, 48], dtype='float32')
y = np.asarray([])

def savepkl(obj, path_out):
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)

    file_save = open(path_out, 'wb')
    pickle.dump(obj, file_save, -1)

    file_save.close()

for direct in os.listdir(video_directory):
    if not os.path.isdir(video_directory+'/'+direct):
        continue
    (data, labels) = VIPerDS.pkl_video(video_directory+'/'+direct, "prova.pkl")
    dataset = np.vstack([dataset, data])
    y = np.append(y, labels)
    if y.size > 30000:
        break

savepkl((dataset, y), "video_dataset.pkl")