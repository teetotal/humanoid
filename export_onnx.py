import argparse
from tensorflow import keras
import os

directory = "onnx"
if not os.path.exists(directory):
    os.makedirs(directory)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False, default="humanoid_model_cnn_h1_d1.5_norm.h5", help="model file path. default humanoid_model_cnn_h1_d1.5_norm.")

args = parser.parse_args()
model_path = args.model
model = keras.models.load_model(model_path)
if model == None:
    print('Error Invalid Model Path')
else:
    path_pb = directory + '/' + model_path + ".pb"
    path_onnx = directory + '/' + model_path + ".onnx"
    model.save(path_pb, save_format='tf')
    os.system('python -m tf2onnx.convert --saved-model ' + path_pb + ' --opset 12 --output ' + path_onnx)