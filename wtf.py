import tables

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

print(y[1000:1040])