import tensorflow as tf


def normalize(data_array, mean_array, std_array):
    return (data_array - mean_array) / std_array


def backscale(data_array, mean_array, std_array):
    return data_array * std_array + mean_array

def create_dataset_from_tensor_slices(x_slice, y_slice, batch_size):
    return tf.data.Dataset.from_tensor_slices(
        (x_slice, y_slice)).batch(batch_size)


def reshape_into_3d(data):
    return data.reshape(len(data), len(data[0, :]), 1)
