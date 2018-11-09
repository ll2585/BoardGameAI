def use_gpu():
    import tensorflow as tf

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)

def write_to_new_file(filename, x, y):
    print("Writing file")
    import tables
    hdf5_file = tables.open_file(filename, 'w')
    filters = tables.Filters(complevel=5, complib='blosc')
    x_storage = hdf5_file.create_earray(hdf5_file.root, 'x',
                                        tables.Atom.from_dtype(x.dtype),
                                        shape=(0, x.shape[-1]),
                                        filters=filters,
                                        expectedrows=len(x))
    y_storage = hdf5_file.create_earray(hdf5_file.root, 'y',
                                        tables.Atom.from_dtype(y.dtype),
                                        shape=(0,),
                                        filters=filters,
                                        expectedrows=len(y))
    for n, (d, c) in enumerate(zip(x, y)):
        x_storage.append(x[n][None])
        y_storage.append(y[n][None])
    hdf5_file.close()

def append_to_file(filename, x, y):
    import tables
    hdf5_path = filename
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='a')
    extendable_hdf5_x = extendable_hdf5_file.root.x
    extendable_hdf5_y = extendable_hdf5_file.root.y
    for n, (d, c) in enumerate(zip(x, y)):
        extendable_hdf5_x.append(x[n][None])
        extendable_hdf5_y.append(y[n][None])
    extendable_hdf5_file.close()
