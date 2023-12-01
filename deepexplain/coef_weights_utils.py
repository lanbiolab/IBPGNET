import sys

import numpy as np
from keras import backend as K
from keras.engine import InputLayer
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from deepexplain.model_utils import get_layers




#second enter
def get_deep_explain_score_layer(model, X, layer_name, output_index=-1, method_name='grad*input'):
    scores = None
    import keras
    from deepexplain.tensorflow_ import DeepExplain
    import tensorflow as tf
    ww = model.get_weights()
    with tf.Session() as sess:
        try:
            with DeepExplain(session=sess) as de:  # <-- init DeepExplain context
                # Need to reconstruct the graph in DeepExplain context, using the same weights.
                # model= nn_model.model
                print(layer_name)
                model = keras.models.clone_model(model)
                model.set_weights(ww)
                # if layer_name=='inputs':
                #     layer_outcomes= X
                # else:
                #     layer_outcomes = nn_model.get_layer_output(layer_name, X)[0]

                x = model.get_layer(layer_name).output
                # x = model.inputs[0]
                if type(output_index) == str:
                    y = model.get_layer(output_index).output
                else:
                    y = model.outputs[output_index]

                # y = model.get_layer('o6').output
                # x = model.inputs[0]
                print(layer_name)
                print('model.inputs', model.inputs)
                print('model y', y)
                print('model x', x)
                attributions = de.explain(method_name, y, x, model.inputs[0], X)
                print('attributions', attributions.shape)
                scores = attributions
                return scores
        except:
            sess.close()
            print("Unexpected error:", sys.exc_info()[0])
            raise




