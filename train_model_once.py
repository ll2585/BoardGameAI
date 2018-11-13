import numpy as np
from ai.ai import AI
import keras
import tables
import constants

constants.use_gpu()

print("Loading data")
BREAK_MOVES = 42
LAST_MODEL = 44
data_file = 'minimax_{0}_to_{1}'.format(BREAK_MOVES, LAST_MODEL)
data_file = 'minimax_50'
hdf5_path = "./data/{0}.hdf5".format(data_file)
extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
x = extendable_hdf5_file.root.x[:]
y = extendable_hdf5_file.root.y[:]
extendable_hdf5_file.close()
print("Data loaded")

latest_version = None

ai = AI()
ai_name = data_file
ai.load_data(x,y)
print("AI Loaded data")
if latest_version is None:
    ai.create_model()
    latest_version = -1
else:
    ai.load_model(ai_name, index=latest_version)
    print("AI Model Loaded")
ai.train_model(x=x, y=y)
ai.save_model(ai_name, index=latest_version+1)