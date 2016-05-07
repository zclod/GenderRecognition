import numpy as np
from scipy import misc

dataset_path = 'D:/PRML/DS/CMD/CMD/CMD/'
video_names = ['1airport1', '1chinacross2', '1chinacross4', '1dawei1', '1dawei5', '1grand1', '1grand3', '1japancross2',
              '1japancross3', '1manko3', '1manko29', '1shatian3', '1thu10', '2dawei1', '2grand6', '2jiansha5',
              '2manko2', '2niurunning2', '3shatian6', 'randomcross3']


def read_image(path):
    img = misc.imread(path, 0)
    return img


def get_id_images(sample_id, video_name):

    # load trajectories_BB.txt for a specific video
    with open(dataset_path + video_name + '/labels.txt') as f:
        # t = f.readlines()
        traj = np.loadtxt(f)

    id_traj = traj[traj[:,1] == sample_id, :]

    sex = id_traj[:, 9]

    imgs = np.empty([0, 3, 128, 48], dtype='float32')
    for s in id_traj:
        img = read_image(dataset_path + video_name + "/%06d.jpg" % s[0])

        img = img[s[6]: s[6] + s[8], s[5]: s[5] + s[7], :]

        img = misc.imresize(img, (128, 48, 3))

        img = np.reshape(img, (1, 3, 128, 48))

        img = np.asarray(img, dtype='float32') / np.max(img)
        imgs = np.vstack([imgs, img])

    return imgs, sex




if __name__ == '__main__':
    imgs, lab = get_id_images(5, video_names[0])
    # import cv2
    # for img in imgs:
    #     img = np.reshape(img, (128, 48, 3)) * 255
    #     cv2.imshow("", img.astype(np.uint8))
    #     cv2.waitKey()