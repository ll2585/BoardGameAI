import numpy as np
from ai.ai import AI
import keras
import tables

data_filename = 'data'

hdf5_path = "./data/{0}.hdf5".format(data_filename)
extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
x = extendable_hdf5_file.root.x[:]
y = extendable_hdf5_file.root.y[:]
extendable_hdf5_file.close()

latest_version = None

ai = AI()
ai.load_data(x,y)
if latest_version is None:
    ai.create_model()
    latest_version = -1
else:
    ai.load_model('model', index=latest_version)
ai.train_model()
ai.save_model('model', index=latest_version+1)