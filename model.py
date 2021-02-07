from keras import models, layers
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Conv2D, Flatten, MaxPooling2D
import keras
#from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout, Dot
import random
import os
import numpy as np
from h5 import *
from hyper_params import *
import tensorflow as tf
from sys import stdout
from datetime import date, time, datetime

config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)

params = get_hyperparams()
##################################################################
def create_model(dim_x, dim_y, dim_h, hiddens, optimizer='Adam', activation='relu'):
    model = models.Sequential()
    model.add(layers.Dense(dim_x, activation=activation, input_shape=(dim_x, )))
    model.add(layers.BatchNormalization())
    for _ in range(hiddens):
        model.add(layers.Dense(dim_h, activation=activation))
        model.add(layers.BatchNormalization())
        
    model.add(layers.Dense(dim_y, activation='softmax'))
    #compile
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #summary
    model.summary()

    return model
##################################################################
def create_model_fit(_x, _y, dim_h, hiddens, epochs, path, optimizer, activation, x_test, y_test, is_batch_norm, dropout) :
    dim_x = len(_x[0])
    dim_y = len(_y[0])
    
    model = models.Sequential()
    model.add(layers.Dense(dim_x, activation=activation, input_shape=(dim_x, )))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.3))
    for _ in range(hiddens):
        model.add(layers.Dense(dim_h, activation=activation))
        if is_batch_norm == True:
            model.add(layers.BatchNormalization())
        elif dropout > 0.0:
            model.add(layers.Dropout(dropout))
        
    model.add(layers.Dense(dim_y, activation='softmax'))
    #compile
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #summary
    model.summary()
    fit(model, _x, _y, epochs, path, x_test, y_test)
    return model
##################################################################
def create_model_cnn_fit(input_shape, output_shape, _x, _y, dim_h, hiddens, epochs, path, optimizer, x_test, y_test) :    
    model = create_model_cnn(input_shape, output_shape, dim_h, hiddens, path, optimizer)
    fit_cnn(model, _x, _y, epochs, path, x_test, y_test)
    return model
##################################################################
def create_model_cnn(input_shape, dim_y, dim_h, hiddens, path, optimizer):
    activation = params['model']['activation']
    is_batch_norm = params['model']['batchnormalization']
    dropout = params['model']['dropout']
    filters = params['model']['cnn']['filters']
    is_pooling = params['model']['cnn']['pooling']
    
    model = models.Sequential(name=path.replace('.h5', ''))
    model.add(Conv2D(filters[0], kernel_size=(2, 2), activation=activation, input_shape=(input_shape[0], input_shape[1], input_shape[2]), name='input'))
    if is_pooling == True:
        model.add(MaxPooling2D(pool_size=(2, 2)))
    filters = filters[1:]
    for f in filters:
        model.add(Conv2D(f, kernel_size=(2, 2), activation=activation))
        if is_pooling == True:
            model.add(MaxPooling2D(pool_size=(2, 2)))

    '''
    if is_batch_norm == True:
            model.add(layers.BatchNormalization())
    elif dropout > 0.0:
        model.add(layers.Dropout(dropout))
    ''' 
    model.add(Flatten())
    for n in range(hiddens):
        model.add(layers.Dense(dim_h, activation=activation, name='hidden_' + str(n)))
        if is_batch_norm == True:
            model.add(layers.BatchNormalization())
        elif dropout > 0.0:
            model.add(layers.Dropout(dropout))
        
    model.add(layers.Dense(dim_y, activation='sigmoid', name='output'))
    #compile
    model.compile(optimizer=optimizer, loss='mse') #, metrics=['accuracy'])
    #summary
    model.summary()
    return model
##################################################################
def fit(model, _x, _y, epochs, path, x_test, y_test):
    #tensorboard
    directory = "tensorboard"
    if not os.path.exists(directory):
        os.makedirs(directory)
    tensorboard_path = path.replace('/', '_')
    tensorboard_path = tensorboard_path.split('.')[0]
    tensorboard_path = directory + '/' + tensorboard_path
    
    freq = 1
    if epochs > 10:
        freq = (int)(epochs / 4)

    tb_hist = keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=freq, write_graph=True, write_images=False)

    #checkpoint
    directory = "checkpoint"
    if not os.path.exists(directory):
        os.makedirs(directory)
    chk = keras.callbacks.ModelCheckpoint(directory + '/'+ path.split('/')[-1])

    #data size
    print("train data size", len(_y))

    ten = int(len(x_test))
    if ten > 10000:
        ten = 10000
    #fit
    if(len(x_test) > 0):
        test_x, test_y = randomize2(x_test, y_test, ten)
        model.fit(_x, _y, batch_size=None, epochs=epochs, validation_data=(test_x, test_y), shuffle=True)
    else:
        model.fit(_x, _y, batch_size=None, epochs=epochs, shuffle=True)
    #save
    model.save(path)
    print('save model', path)
    #evaluate
    if(len(x_test) > 0):
        print("evaluate")
        ten = int(len(_x))
        if ten > 10000:
            ten = 10000
        test_x, test_y = randomize2(_x, _y, ten)
        e = evaluate(model, test_x, test_y)
        print(e)
##################################################################
def fit_cnn(model, _x, _y, epochs, path, x_test, y_test):

    callbacks = []

    if params['model']['tensorboard']['enable'] == True:

        #tensorboard
        directory = "tensorboard"
        if not os.path.exists(directory):
            os.makedirs(directory)
        tensorboard_path = path.replace('/', '_')
        tensorboard_path = tensorboard_path.replace('.h5', '')
        tensorboard_path = directory + '/' + tensorboard_path
       
        '''
        write_graph = params['model']['tensorboard']['write_graph']
        tb_hist = keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=freq, write_graph=write_graph, write_images=False)
        callbacks.append(tb_hist)
        '''

    if params['model']['checkpoint'] == True:
        #checkpoint
        directory = "checkpoint"
        if not os.path.exists(directory):
            os.makedirs(directory)
        chk = keras.callbacks.ModelCheckpoint(directory + '/'+ path.split('/')[-1])
        callbacks.append(chk)

    #data size
    print("train data size", len(_y))

    ten = int(len(x_test))
    if ten > 10000:
        ten = 10000
    #fit
    if len(x_test) > 0:
        test_x, test_y = randomize2(x_test, y_test, ten)
        model.fit(_x, _y, batch_size=None, epochs=epochs, callbacks=callbacks, validation_data=(test_x, test_y), shuffle=True)
    else:
        model.fit(_x, _y, batch_size=None, epochs=epochs, callbacks=callbacks, shuffle=True)
    #save
    model.save(path)
    print('save model', path)
    #evaluate
    if len(x_test) > 0:
        print("evaluate")
        ten = int(len(_x))
        if ten > 10000:
            ten = 10000
        test_x, test_y = randomize2(_x, _y, ten)
        e = evaluate(model, test_x, test_y)
        print(e)
##################################################################
def evaluate(model, _x, _y):
    return model.evaluate(_x, _y)
##################################################################
def train_on_batch(model, x_arr, y_arr, text = ''):
    #return model.train_on_batch(x_arr, y_arr)
    batch_size = params['model']['batch']
    size = len(x_arr)
    freq = int(size / batch_size)
    if freq < 1000:
        freq = 1
    else:
        freq = int(freq / 1000)

    idx = 0
    loss = 0.0
    acc = 0.0
    cnt = 0

    min_loss = 0.0
    min_x, min_y = [], []
    min_idx = -1
    min_end = -1

    while idx < size:
        end = idx + batch_size
        if end >= size:
            end = size
        x = x_arr[idx:end]
        y = y_arr[idx:end]
        ret = model.train_on_batch(x, y)

        if ret[0] > min_loss:
            min_loss = ret[0]
            min_x = x
            min_y = y
            min_idx = idx
            min_end = end

        idx += batch_size
        loss += ret[0]
        acc += ret[1]

        cnt += 1

        if cnt % freq == 0 or end == size:
            sz_loss = '%.4f' % float(loss / cnt)
            sz_acc = '%.4f' % float(acc / cnt)
            sz = ' loss: ' + sz_loss + ' accuracy: ' + sz_acc
            stdout.write('\r' + str(datetime.now()) + ' ' + text + ' ' + str(end) + '/' + str(size) + sz)
            stdout.flush()
    
    print()
    if(len(min_x) > 0 and len(min_y) > 0):
        arr = []
        for _ in range(4):
            ret = model.train_on_batch(min_x, min_y)
            arr.append(ret[0])
        
        print(str(datetime.now()), 'min idx', min_idx, '~', min_end, 'loss', min_loss, '->', arr)

    return [loss / cnt, acc / cnt]
    
##################################################################
def save_model(model, path):
    model.save(path)
##################################################################
def load_model(path):
    if(os.path.isfile(path) == False):
        print('no exists', path)
        return None
    else:
        model = models.load_model(path)
        print('loaded model', path)
        return model
##################################################################
def prediction_argmax(model, _x, _y = None, is_evaluate = True, _meta = None):
    ret = model.predict(_x)

    if is_evaluate == True:
        pred = np.argmax(ret, axis=1)
        y = np.argmax(_y, axis=1)
        
        correct = 0
        for n in range(len(pred)):
            if pred[n] == y[n]:
                correct += 1
            else:
                print('wrong prediction', pred[n], 'Y', y[n], _meta[n])
                debug(_meta[n], _x[n], y[n])

        print('Accuracy', correct / len(_y), correct, len(_y))
    
    return ret
##################################################################
def get_data(path, is_print = True):
    h5 = h5_read(path)
    _x = h5['/train']['X'][()]
    _y = h5['/train']['Y'][()]    
    
    if is_print == True:
        print("loaded data", path, len(_x))
    
    '''
    print("< sample train data >")
    debug(META[0], X[0], Y[0])
    '''

    return _x, _y
##################################################################
def read_dir(_path):
    files = os.listdir(_path)
    files = [f for f in files if f.endswith(".h5")]
    files.sort()

    return files
##################################################################
def load_data(_path, is_flatten = False):
    files = read_dir(_path)

    x = []
    y = []
    
    
    for f in files:
        _x, _y = get_data(_path + "/" + f)
        if is_flatten == True:
            x_arr = []
            for e in _x:
                x_arr.append(np.array(e).flatten())
            
            _x = x_arr

        x.extend(_x)
        y.extend(_y)        

    x = np.array(x)
    y = np.array(y)    

    return x, y
##################################################################
def randomize2(arr1, arr2, size):
    total = len(arr1)
    r = np.random.choice(total, size, replace=False)
    array1 = []
    array2 = []
    for e in r:
        array1.append(arr1[e])
        array2.append(arr2[e])
    
    return np.array(array1), np.array(array2)
##################################################################
def randomize3(arr1, arr2, arr3, size):
    total = len(arr1)
    r = np.random.choice(total, size, replace=False)
    array1 = []
    array2 = []
    array3 = []
    for e in r:
        array1.append(arr1[e])
        array2.append(arr2[e])
        array3.append(arr3[e])
    
    return np.array(array1), np.array(array2), np.array(array3)
##################################################################
def get_weights(model):
    configs = []
    weights = []
    for layer in model.layers:
        w = layer.get_weights()
        c = layer.get_config()
        configs.append(c)
        weights.append(w)
        print(c['name'])
    return weights, configs