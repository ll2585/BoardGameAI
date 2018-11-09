import os
import tables
import numpy as np
from ai.ai import AI

directory = os.fsencode('./data')

all_x = []
all_y = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".hdf5"):
        file_path = './data/{0}'.format(filename)
        extendable_hdf5_file = tables.open_file(file_path, mode='r')
        x = extendable_hdf5_file.root.x[:]
        y = extendable_hdf5_file.root.y[:]
        extendable_hdf5_file.close()
        all_x.append(x)
        all_y.append(y)
        break
    else:
        continue

xs = np.concatenate(all_x)
ys = np.concatenate(all_y)

ai = AI()
ai.load_data(xs,ys)
print("AI Loaded data")
ai.create_model()
latest_version = -1
ai.train_model()
ai.save_model('all_data', index=9999)
