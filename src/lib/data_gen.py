import os
import pickle

import numpy as np
import tensorflow as tf
from mlreflect.data_generation import ReflectivityGenerator
from mlreflect.models import DefaultTrainedModel
from mlreflect.training import InputPreprocessor, OutputPreprocessor
from silx.io.dictdump import h5todict
from sklearn.model_selection import train_test_split

from lib import helper

DEBUG = False

if DEBUG:
    N_TRAIN = 1280
    N_TEST = 512
    LOAD_PATH = os.path.join("..", "data", "debug")

else:
    N_TRAIN = 1024000
    N_TEST = 16384
    LOAD_PATH = os.path.join("..", "data")
noise_scales = [np.round(x, 2) for x in np.arange(0, 1.0, 1.0 / 20)]


def save_training_data_and_stats(n_samples_train):
    q_values = DefaultTrainedModel().q_values
    sample = DefaultTrainedModel().sample
    generator = ReflectivityGenerator(q_values, sample)
    labels = generator.generate_random_labels(n_samples_train)
    reflectivity = generator.simulate_reflectivity(labels)
    labels = labels.drop(
        labels.columns.difference(["Film_thickness", "Film_roughness", "Film_sld"]),
        axis=1,
    )

    mean_labels, std_labels = np.mean(labels, axis=0), np.std(labels, axis=0)
    np.savetxt(os.path.join(LOAD_PATH, "labels_real_scale_forvisual.csv"), labels)
    labels = (labels - mean_labels) / std_labels

    np.savetxt(os.path.join(LOAD_PATH, "reflectivity_real_scale.csv"), reflectivity)
    np.savetxt(os.path.join(LOAD_PATH, "labels_unit_scale.csv"), labels)

    np.savetxt(os.path.join(LOAD_PATH, "mean_labels.csv"), mean_labels)
    np.savetxt(os.path.join(LOAD_PATH, "std_labels.csv"), std_labels)
    for noise_scale in noise_scales:
        reflectivity_noise = reflectivity * tf.random.uniform(
            shape=reflectivity.shape, minval=1 - noise_scale, maxval=1 + noise_scale
        )
        mean_data, std_data = (
            np.mean(reflectivity_noise, axis=0),
            np.std(reflectivity_noise, axis=0),
        )

        np.savetxt(os.path.join(LOAD_PATH, f"mean_data_{noise_scale}.csv"), mean_data)
        np.savetxt(os.path.join(LOAD_PATH, f"std_data_{noise_scale}.csv"), std_data)


def save_test_data_and_stats(n_samples_test):
    q_values = DefaultTrainedModel().q_values
    sample = DefaultTrainedModel().sample
    generator = ReflectivityGenerator(q_values, sample)
    labels = generator.generate_random_labels(n_samples_test)
    ip = InputPreprocessor()
    out = OutputPreprocessor(sample)
    reflectivity_real_scale = generator.simulate_reflectivity(labels)

    labels_full = labels.copy()
    labels = labels.drop(
        labels.columns.difference(["Film_thickness", "Film_roughness", "Film_sld"]),
        axis=1,
    )
    labels = labels.to_numpy()
    for noise_scale in noise_scales:
        uniform_noise_range_low = 1 - noise_scale
        uniform_noise_range_high = 1 + noise_scale
        reflectivity_real_scale = reflectivity_real_scale * np.random.uniform(
            uniform_noise_range_low,
            uniform_noise_range_high,
            reflectivity_real_scale.shape,
        )
        np.savetxt(
            os.path.join(LOAD_PATH, f"test_reflectivity_real_scale_{noise_scale}.csv"),
            reflectivity_real_scale,
        )

    mean_labels, std_labels = (
        np.loadtxt(os.path.join(LOAD_PATH, "mean_labels.csv")),
        np.loadtxt(os.path.join(LOAD_PATH, "std_labels.csv")),
    )
    labels = (labels - mean_labels) / std_labels
    np.savetxt(os.path.join(LOAD_PATH, "test_labels_unit_scale.csv"), labels)
    with open(os.path.join(LOAD_PATH, "test_data_objects"), "wb") as outp:
        pickle.dump(
            [generator, ip, out, q_values, labels_full], outp, pickle.HIGHEST_PROTOCOL
        )


def iterate_experiments():
    test_refl_lst = []
    test_q_values_lst = []
    lables_lst = []
    pen2 = h5todict("evaluation/xrr_dataset.h5")
    for _, value_dict in pen2.items():
        test_refl = value_dict["experiment"]["data"]
        test_q_values = value_dict["experiment"]["q"]
        true_labels_dict = value_dict["fit"]
        lables = [
            true_labels_dict["Film_thickness"],
            true_labels_dict["Film_roughness"],
            np.real(true_labels_dict["Film_sld"]),
        ]
        test_refl_lst.append(test_refl)
        test_q_values_lst.append(test_q_values)
        lables_lst.append(lables)
    return test_refl_lst, test_q_values_lst, lables_lst


def create_train_val_test_datasets(dataloader, hp_config):
    # refact
    train_data = helper.reshape_into_3d(dataloader.data_train_real_scale)
    test_data = helper.reshape_into_3d(dataloader.data_test_real_scale)

    x_train, x_val, y_train, y_val = train_test_split(
        train_data, dataloader.labels_train_unit_scale, test_size=0.2
    )
    train_dataset_real_scale = helper.create_dataset_from_tensor_slices(
        x_slice=x_train, y_slice=y_train, batch_size=hp_config.BATCH_SIZE
    )
    test_dataset_real_scale = helper.create_dataset_from_tensor_slices(
        x_slice=test_data,
        y_slice=dataloader.labels_test_unit_scale,
        batch_size=hp_config.BATCH_SIZE,
    )

    x_val_noisy = x_val * np.random.uniform(
        low=1 - hp_config.noise_level, high=1 + hp_config.noise_level, size=x_val.shape
    )
    validation_dataset_real_scale = tf.data.Dataset.from_tensor_slices(
        (x_val_noisy, y_val)
    ).batch(hp_config.BATCH_SIZE)

    return (
        train_dataset_real_scale,
        test_dataset_real_scale,
        validation_dataset_real_scale,
    )


def main():
    save_training_data_and_stats(n_samples_train=N_TRAIN)
    save_test_data_and_stats(n_samples_test=N_TEST)


if __name__ == "__main__":
    main()
