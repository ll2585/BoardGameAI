from ai.ai import AI
import tables
from pprint import pprint

def load_ai(main_name, index):
    ai = AI()
    ai.load_model(main_name, index=index)
    return ai

def load_data(filename):
    hdf5_path = "./data/{0}.hdf5".format(filename)
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    x = extendable_hdf5_file.root.x[:]
    y = extendable_hdf5_file.root.y[:]
    extendable_hdf5_file.close()
    return x, y

def evaluate_ai(ai, x, y):
    return ai.evaluate_data(x,y)

latest_version = 12

model_accuracies = {}
for model_version in range(latest_version+1):
    model_accuracies[model_version] = {}
    ai = load_ai('model', model_version)
    for i in range(5):
        x, y = load_data('data_new_{0}'.format(i))
        model_accuracies[model_version][i] = evaluate_ai(ai, x, y)

pprint(model_accuracies)