import pickle as cPickle
import logging
import os
import time


from keras.models import Sequential
from matplotlib import pyplot as plt




def print_model(model, level=1):
    for i, l in enumerate(model.layers):
        indent = '  ' * level + '-'
        if type(l) == Sequential:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))
            print_model(l, level + 1)
        else:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))


#use
def get_layers(model, level=1):
    layers = []
    for i, l in enumerate(model.layers):

        # indent = '  ' * level + '-'
        if type(l) == Sequential:
            layers.extend(get_layers(l, level + 1))
        else:
            layers.append(l)

    return layers


from deepexplain.coef_weights_utils import get_deep_explain_scores
import numpy as np




def get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=True, **kwargs):
   

    if feature_importance.startswith('deepexplain'):
        method = feature_importance.split('_')[1]
        coef_ = get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed,
                                        **kwargs)
   

    else:
        coef_ = None
    return coef_



