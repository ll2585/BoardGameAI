import numpy as np
from ai.ai import AI
import tables

hdf5_path = "./data/data.hdf5"
extendable_hdf5_file = tables.openFile(hdf5_path, mode='r')
x = extendable_hdf5_file.root.x[:]
y = extendable_hdf5_file.root.y[:]
extendable_hdf5_file.close()

latest_version = None

ai = AI()
ai.load_data(x,y)
if latest_version is None:
    ai.create_model()
else:
    ai.load_model('model', index=latest_version)
ai.train_model()
ai.save_model('model', index=latest_version+1)