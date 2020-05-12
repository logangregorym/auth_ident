import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Conv1D, Flatten, BatchNormalization, Lambda
from tensorflow.keras import backend as K

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return [shape1[0], 1]


class contrastive_cnn():

    def __init__(self):

        self.name = "contrastive_cnn"
        self.dataset_type = "split"

    def create_cnn(self, params, index):
        input = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                            params[index]['dataset'].len_encoding),
                            name='place_holder_input')

        conv = Conv1D(128, 63, strides=1, padding="same", activation="relu", name='conv_1')(input)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 31, strides=2, padding="same", activation="relu", name='conv_2')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 30, strides=2, padding="same", activation="relu", name='conv_3')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 15, strides=1, padding="same", activation="relu", name='conv_4')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 7, strides=1, padding="same", activation="relu", name='conv_5')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)

        conv = Flatten()(conv)

        return keras.Model(input, conv)

        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding),
                             name='input_1')
        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding),
                             name='input_2')

        cnn = self.create_cnn(params, index)

        cnn1 = cnn(input1)
        cnn2 = cnn(input2)

        output_embedding = Dense(512, name="output_embedding")

        output_embedding1 = output_embedding(cnn1)
        output_embedding2 = output_embedding(cnn2)

        distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape, name='distance')([output_embedding1, output_embedding2])


        model = keras.Model(inputs=(input1, input2), outputs=distance, name=self.name + "-" + str(index))
        model.summary()

        return model
