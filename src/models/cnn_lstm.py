import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Conv1D

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class cnn_lstm():

    def __init__(self):

        self.name = "cnn_lstm"
        self.dataset_type = "split"
        self.input_embedding_size = 32
        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_1')
        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_2')

        conv1 = Conv1D(128, 150, strides=49, name='hidden_1a')(input1)
        conv1 = Conv1D(256, 117, strides=117, name='hidden_1b')(conv1)
        conv1 = keras.backend.squeeze(conv1, axis=1)

        conv2 = Conv1D(128, 150, strides=49, name='hidden_2a')(input2)
        conv2 = Conv1D(256, 117, strides=117, name='hidden_2b')(conv2)
        conv2 = keras.backend.squeeze(conv2, axis=1)

        lstm1 = LSTM(256, name='lstm1')(input1, initial_state=[conv1, conv1])
        lstm2 = LSTM(256, name='lstm2')(input2, initial_state=[conv2, conv2])

        concat = layers.concatenate([lstm1, lstm2])
        outputs = Dense(1, activation='sigmoid', name='predictions')(concat)

        model = keras.Model(inputs=(input1, input2), outputs=outputs, name=self.name + "-" + str(index))
        model.summary()

        return model
