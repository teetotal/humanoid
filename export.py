import json
from h5 import *
import os

def get_data(data, keys):
    arr = []
    for key in keys:
        arr.append(data[key])
    return arr

with open('C:/Users/daejung/AppData/LocalLow/DefaultCompany/BoneTest/data.json') as json_file:
    json_data = json.load(json_file)

print(len(json_data['data']))

x_keys = []
y_keys = []

for e in json_data['data'][0]['X']:    
    x_keys.append(e)
for e in json_data['data'][0]['Y']:    
    y_keys.append(e)

print(x_keys)    
print(y_keys)

X = []
Y = []

for data in json_data['data']:
    #X
    X.append(get_data(data['X'], x_keys))
    #Y
    Y.append(get_data(data['Y'], y_keys))

print(X)
print(Y)

path_dir = 'data'
if not os.path.exists(path_dir):
    os.makedirs(path_dir)

path = path_dir + '/train-001.h5'
h5 = h5_gen(path, 'gzip')
h5.write('/train', 'X', X)
h5.write('/train', 'Y', Y)

print("Done.", path, len(X), len(Y))