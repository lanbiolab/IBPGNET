import sys

import numpy as np
from keras import backend as K
from keras.engine import InputLayer
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from deepexplain.model_utils import get_layers


#frist enter
def get_deep_explain_scores(model, X_train, y_train, target=-1, method_name='grad*input', detailed=False, **kwargs):
    # gradients_list = []
    # gradients_list_sample_level = []

    gradients_list = {}
    gradients_list_sample_level = {}

    i = 0
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )

            if target is None:
                output = i
            else:
                output = target

            print('layer # {}, layer name {},  output name {}'.format(i, l.name, output))
            i += 1
            gradients = get_deep_explain_score_layer(model, X_train, l.name, output, method_name=method_name)
            if gradients.ndim > 1:
                # feature_weights = np.sum(np.abs(gradients), axis=-2)
                # feature_weights = np.sum(gradients, axis=-2)
                print('gradients.shape', gradients.shape)
                # feature_weights = np.abs(np.sum(gradients, axis=-2))
                feature_weights = np.sum(gradients, axis=-2)
                # feature_weights = np.mean(gradients, axis=-2)
                print('feature_weights.shape', feature_weights.shape)
                print('feature_weights min max', min(feature_weights), max(feature_weights))
            else:
                # feature_weights = np.abs(gradients)
                feature_weights = gradients

            gradients_list[l.name] = feature_weights
            gradients_list_sample_level[l.name] = gradients
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list
    pass



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




