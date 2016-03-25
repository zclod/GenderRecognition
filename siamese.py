__author__ = 'marco'
from keras.models import Sequential
from keras.layers.core import Dense, Siamese

def build_siamese(input_model_1, input_model_2, input_dim, output_dim):
    """

    :param input_model_1:
    :type input_model_1:
    :param input_model_2:
    :type input_model_2:
    :param input_dim: last layer input
    :type input_dim:
    :param output_dim: last layer output
    :type output_dim:
    :return:
    :rtype:
    """

    inputs = [input_model_1, input_model_2]

    layer = Dense(input_dim=input_dim, output_dim=output_dim)

    model = Sequential()
    # mode: one of {sum, mul, concat, ave, join, cos, dot}.
    model.add(Siamese(layer, inputs, 'sum'))

    # model.compile(loss='mse', optimizer='sgd')
    return model


