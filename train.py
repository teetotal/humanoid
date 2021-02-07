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
parser.add_argument('--epochs', type=int, required=False, default=128, help="epochs. default 128.")
parser.add_argument('--hiddens', type=int, required=False, default=1, help="count of hidden layer. default 1.")
parser.add_argument('--hidden_dim', type=float, required=False, default=1.5, help="hidden layer dim = input dim x (hiddens_dim). default 1.5.")
args = parser.parse_args()
_path = args.path
_test = args.test
_epochs = args.epochs
hiddens = args.hiddens
hidden_dim = args.hidden_dim
model_path = args.model

if model_path == None:
    if params['model']['batchnormalization'] == True:
        sz = "_norm"
    elif params['model']['dropout'] > 0.0:
        #sz = "_drop" + str(int(params['model']['dropout'] * 100))
        sz = "_drop" + str(params['model']['dropout'])
    model_path = "humanoid_model_cnn_h" + str(hiddens) + "_d" + str(hidden_dim) + sz + ".h5"

#15
shape = [5, 3, 3]
dim_output = 20 * 3

print('Loading train data ...')
X, Y = load_data(_path)
X = X.reshape(len(X), shape[0], shape[1], shape[2])
Y = Y.reshape(len(Y), dim_output)

print('Loading test data ...')
x_test, y_test = load_data(_test)
x_test = x_test.reshape(len(x_test), shape[0], shape[1], shape[2])
y_test = y_test.reshape(len(y_test), dim_output)

dim_h = int(shape[0] * shape[1] * shape[2] * hidden_dim)

import keras.optimizers
model = load_model(model_path)
if model == None:
    adam = keras.optimizers.Adam(lr=params['model']['lr'])
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