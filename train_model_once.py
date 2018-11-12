import numpy as np
from ai.ai import AI
import keras
import tables
import constants

constants.use_gpu()

print("Loading data")
hdf5_path = "./data/seeded.hdf5"
extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
x = extendable_hdf5_file.root.x[:]
y = extendable_hdf5_file.root.y[:]
extendable_hdf5_file.close()
print("Data loaded")

latest_version = None

ai = AI()
ai_name = 'seeded_last_20'
ai.load_data(x,y)
print("AI Loaded data")
if latest_version is None:
    ai.create_model()
    latest_version = -1
else:
    ai.load_model(ai_name, index=latest_version)
    print("AI Model Loaded")
x, y = ai.filter_by_tiles_collected(20)
ai.train_model(x=x, y=y)
ai.save_model(ai_name, index=latest_version+1)