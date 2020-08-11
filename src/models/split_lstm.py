import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class split_lstm():

    def __init__(self):

        self.name = "split_lstm"
        self.dataset_type = "split"
        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding),
                             name='input_1')

        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding),
                             name='input_2')

        dense_1 = Dense(32, name="dense_1")
        lstm_1 = LSTM(512, name='output_embedding')

        dense1 = dense_1(input1)
        dense2 = dense_1(input2)

        lstm1 = lstm_1(dense1)
        lstm2 = lstm_1(dense2)

        output_embedding = layers.concatenate([lstm1, lstm2], name="concatenate")
        print(str(output_embedding))
        outputs = Dense(1, activation='sigmoid', name='predictions')(output_embedding)

        return keras.Model(inputs=[input1, input2], outputs=outputs, name=self.name + "-" + str(index))




