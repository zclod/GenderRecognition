import numpy
import scipy.io
from PIL import Image
import glob
import os.path
from os import listdir

directory = '/home/cla/Downloads/VIPeR.v1.0/viper.v1.0'
video_directory = '/home/cla/Downloads/CMD/CMD/CMD'
ds_directory = directory + '/VIPeR'
file_mat = directory + '/Viper_attributes_1/bmvc2012_VIPeR_attribute_annotations.mat'
dataset_path = "dataset.pkl"


def __load_data1(train_size=316):
    """
    Load dataset:
        Build training set picking randomly sample from both cam_a and cam_b
        If a sample is chosen (its index is selected) both images (cam_a and cam_b) are added to training set;
        the remaining samples represent the testing set

    :return:
    :rtype:
    """
    # number of person in the dataset
    N = 632
    if train_size > N:
        raise IndexError("train_size is greater than dataset size")

    labels = __load_labels()
    (cam_a, cam_b) = __load_images()

    # test_size = N - train_size

    # choice = Generates a random sample from a given 1-D array
    # set training example, get random index (and then sort) with no replacement
    # array 1-D of 316 elem in range 0-632
    train_ind = numpy.sort(numpy.random.choice(N, train_size, replace=False))
    # get testing example as the remaining index
    test_ind = numpy.setdiff1d(range(N), train_ind)

    X_train = numpy.empty([0, 3, 128, 48], dtype='float32')
    y_train = []

    X_test = numpy.empty([0, 3, 128, 48], dtype='float32')
    y_test = []

    for index in train_ind:
        img_train_a = cam_a[index].reshape(1, 3, 128, 48)
        img_train_b = cam_b[index].reshape(1, 3, 128, 48)

        X_train = numpy.vstack([X_train, img_train_a])
        X_train = numpy.vstack([X_train, img_train_b])

        y_train.append(labels[index])
        y_train.append(labels[index])

    for index in test_ind:
        img_train_a = cam_a[index].reshape(1, 3, 128, 48)
        img_train_b = cam_b[index].reshape(1, 3, 128, 48)

        X_test = numpy.vstack([X_test, img_train_a])
        X_test = numpy.vstack([X_test, img_train_b])

        y_test.append(labels[index])
        y_test.append(labels[index])

    y_train = numpy.asarray(y_train, dtype='int8')
    y_test = numpy.asarray(y_test, dtype='int8')

    return (X_train, y_train), (X_test, y_test)


def __load_data2():
    """
    Load dataset (cam_a and cam_b):
        Build training set picking randomly a sample from cam_a and cam_b (alternatively)
        If cam_a image is chosen the corresponding cam_b image goes to testing set
    :return:
    :rtype:
    """

    labels = __load_labels()
    (cam_a, cam_b) = __load_images()

    # number of person in the dataset
    N = 632
    train_size = N

    # array with N index {0, 1}: 0=cam_a  1=cam_b
    train_ind = numpy.random.choice([0, 1], train_size, replace=True)
    # test_ind = numpy.ones((N,), dtype='int32')
    # test_ind = test_ind-train_ind

    X_train = numpy.empty([0, 3, 128, 48], dtype='float32')
    # y_train = numpy.empty([0,], dtype='float32')

    X_test = numpy.empty([0, 3, 128, 48], dtype='float32')
    # y_test = numpy.empty([0,], dtype='float32')

    y_train = y_test = labels

    for i in range(N):
        if train_ind[i] == 0:
            img_train = cam_a[i]
            img_test = cam_b[i]
        else:
            img_train = cam_b[i]
            img_test = cam_a[i]

        img_train = img_train.reshape(1, 3, 128, 48)
        img_test = img_test.reshape(1, 3, 128, 48)

        X_train = numpy.vstack([X_train, img_train])
        X_test = numpy.vstack([X_test, img_test])

    return (X_train, y_train), (X_test, y_test)


def __load_data3():
    """
    Load dataset: use images from cam_b as training set
                  and images from cam_a as testing set
    :return:
    :rtype:
    """

    labels = __load_labels()
    (cam_a, cam_b) = __load_images()

    X_train = cam_b
    X_test = cam_a

    y_train = y_test = labels

    return (X_train, y_train), (X_test, y_test)


def __load_data4():
    """
    Load dataset: use images from cam_b as training set
                  and images from cam_a as testing set
    :return:
    :rtype:
    """

    labels = __load_labels()
    (cam_a, cam_b) = __load_images()

    X_train = cam_b
    X_test = cam_a

    y_test = y_train = labels


    # (img_video, video_label) = __load_images_video()

    # y_train = numpy.append(labels, video_label)
    # X_train = numpy.vstack([X_train, img_video])

    return (X_train, y_train), (X_test, y_test)


def __read_image(address):
    img = Image.open(open(address))
    # put image in a numPy array
    img = numpy.asarray(img, dtype='float32') / 256.
    # put image in 4D tensor of shape (1, 3, height, width)
    img = img.transpose(2, 0, 1).reshape(1, 3, 128, 48)
    return img


def __load_images():

    imgfiles1 = sorted(glob.glob(ds_directory + '/cam_a/*.bmp'))
    imgfiles2 = sorted(glob.glob(ds_directory + '/cam_b/*.bmp'))

    img_data_a = numpy.empty([0, 3, 128, 48], dtype='float32')
    for img_address in imgfiles1:
        img = __read_image(img_address)
        # same MATLAB operation A = [A ; newA] add a row
        img_data_a = numpy.vstack([img_data_a, img])

    img_data_b = numpy.empty([0, 3, 128, 48], dtype='float32')
    for img_address in imgfiles2:
        img = __read_image(img_address)
        img_data_b = numpy.vstack([img_data_b, img])

    return img_data_a, img_data_b


def __load_images_video():

    img_data_a = numpy.empty([0, 3, 128, 48], dtype='float32')
    label = numpy.asarray([])

    for direct in listdir(video_directory):
        if not os.path.isdir(video_directory+'/'+direct):
            continue

        imgfiles = sorted(glob.glob(video_directory+'/'+direct + '/images/*.bmp'))
        lab = numpy.loadtxt(video_directory+'/'+direct + '/images/sex')

        label = numpy.append(label, lab)

        for img_address in imgfiles:
            img = __read_image(img_address)
            # same MATLAB operation A = [A ; newA] add a row
            img_data_a = numpy.vstack([img_data_a, img])

    return img_data_a, label

def pkl_video(dir, file_out=None):
    imgfiles = sorted(glob.glob(dir + '/images/*.bmp'))
    lab = numpy.loadtxt(dir + '/images/sex')

    img_data_a = numpy.empty([lab.size, 3, 128, 48], dtype='float32')

    i=0
    for img_address in imgfiles:
        img = __read_image(img_address)
        # same MATLAB operation A = [A ; newA] add a row
        img_data_a[i] = img
        i+=1

    #__save((img_data_a, lab), dir + "/" + file_out)
    return  (img_data_a, lab)


def __load_labels():

    mat = scipy.io.loadmat(file_mat)

    # labels = numpy.asarray(mat['A'], dtype=numpy.dtype('b'))
    labels = numpy.asarray(mat['A'], dtype=numpy.dtype(numpy.int8))

    # 0=donna
    # 1=uomo
    gender = labels[:, 13]
    idx = (gender == 0)

    gender[idx] = -1

    return gender


def __save(obj, path_out):
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)

    file_save = open(path_out, 'wb')
    pickle.dump(obj, file_save, -1)

    file_save.close()


def __load(path_in):
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)

    file_load = open(path_in, 'rb')
    m = pickle.load(file_load)
    return m


def __save_pickeData():
    (X_train, y_train), (X_test, y_test) = load_dataset()
    __save(((X_train, y_train), (X_test, y_test)), dataset_path)
    return (X_train, y_train), (X_test, y_test)


def load_dataset(ds=4):
    if ds == 1:
        return __load_data1()
    if ds == 2:
        return __load_data2()
    if ds == 3:
        return __load_data3()
    if ds == 4:
        return __load_data4()
    raise StandardError("ds not in range 1-3")


def load_data():
    if not os.path.isfile(dataset_path):
        return __save_pickeData()

    return __load_pdata()


def __load_pdata():
    m = __load(dataset_path)
    X_train = m[0][0]
    y_train = m[0][1]

    X_test = m[1][0]
    y_test = m[1][1]

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    # load_data()
    __save_pickeData()
    # __load_pdata()


