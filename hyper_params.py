import json
#####################################

def get_hyperparams():
    json_data = ""
    with open('hyper_params.json') as json_file:
        json_data = json.load(json_file)

    return json_data