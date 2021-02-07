from h5 import *
import argparse
from model import *
import numpy as np
import random
import os
from hyper_params import *

params = get_hyperparams()

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=False, default="data", help="train data file path. default data.")
parser.add_argument('--test', type=str, required=False, default="test", help="test data file path. default test.")
parser.add_argument('--model', type=str, required=False, default=None, help="model file path. default None")
parser.add_argument('--epochs', type=int, required=False, default=512, help="epochs. default 128.")
parser.add_argument('--hiddens', type=int, required=False, default=2, help="count of hidden layer. default 1.")
parser.add_argument('--hidden_dim', type=float, required=False, default=2, help="hidden layer dim = input dim x (hiddens_dim). default 1.5.")
args = parser.parse_args()
_path = args.path
_test = args.test
_epochs = args.epochs
hiddens = params['model']['hiddens']
hidden_dim = params['model']['hidden_dim']
model_path = args.model

if model_path == None:
    '''
    sz = ""
    if params['model']['batchnormalization'] == True:
        sz = "_norm"
    elif params['model']['dropout'] > 0.0:
        #sz = "_drop" + str(int(params['model']['dropout'] * 100))
        sz = "_drop" + str(params['model']['dropout'])
    model_path = "humanoid_model_cnn_h" + str(hiddens) + "_d" + str(hidden_dim) + sz + ".h5"
    '''
    model_path = "humanoid_model.h5"


dim_output = 20 * 3

print('Loading train data ...')
X, Y = load_data(_path)


print('Loading test data ...')
x_test, y_test = load_data(_test)

import keras.optimizers
model = load_model(model_path)
adam = keras.optimizers.Adam(lr=params['model']['lr'])

if(params['model']['cnn_enable'] == True):
    shape = [3, 12, 3]
    X = X.reshape(len(X), shape[0], shape[1], shape[2])
    Y = Y.reshape(len(Y), dim_output)

    x_test = x_test.reshape(len(x_test), shape[0], shape[1], shape[2])
    y_test = y_test.reshape(len(y_test), dim_output)

    dim_h = int(shape[0] * shape[1] * shape[2] * hidden_dim)
    if model == None:
        model = create_model_cnn_fit(
            shape
            , dim_output
            , X
            , Y
            , dim_h
            , hiddens
            , _epochs
            , model_path
            , adam
            , x_test, y_test
        )
    else: 
        fit_cnn(model, X, Y, _epochs, model_path, x_test, y_test)
else:
    if model == None:
        shape = [3, 12, 3]
        X = X.reshape(len(X), shape[0] * shape[1] * shape[2])
        Y = Y.reshape(len(Y), dim_output)

        x_test = x_test.reshape(len(x_test), shape[0] * shape[1] * shape[2])
        y_test = y_test.reshape(len(y_test), dim_output)

        dim_h = int(shape[0] * shape[1] * shape[2] * hidden_dim)

        model = create_model_fit(X, Y, dim_h, hiddens, _epochs, model_path
        , adam, params['model']['activation']
        , x_test, y_test
        , params['model']['batchnormalization']
        , params['model']['dropout']
        )
    else: 
        fit(model, X, Y, _epochs, model_path, x_test, y_test)