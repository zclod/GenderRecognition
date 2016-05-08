from GetImages import get_id_images
from siamese_model import build_model
from keras.utils import np_utils
import json
import numpy as np

with open("directories.json", "r") as f:
    dirs = json.load(f)

model = build_model((3, 128, 48), "my_model_weights7256l54.h5")

class_ok = 0
total_count = 0

for d in dirs["test"]:
    traj = np.loadtxt(d + "/labels.txt")
    max_id = max(traj[:,1])
    for i in range(1, int(max_id) + 1):
        imgs, labels = get_id_images(i, d)
        if len(imgs) and labels[0] != 0 :
            labels[labels == -1] = 0
            predictions = model.predict_classes(imgs)
            img_class = int((sum(predictions) / float(len(predictions))) >= 0.5)

            total_count += 1
            if img_class == labels[0]:
                class_ok += 1

    print("predictions ok : " + str(class_ok))
    print("total samples : " + str(total_count))
    #print("accuracy : " + str(class_ok / float(total_count)))


# print("predictions ok : " + str(class_ok))
# print("total samples : " + str(total_count))
# print("accuracy : " + str(class_ok / float(total_count)))
