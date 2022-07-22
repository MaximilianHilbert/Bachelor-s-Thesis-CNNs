import numpy as np
from mlreflect.data_generation import ReflectivityGenerator
from mlreflect.models import DefaultTrainedModel
from mlreflect.training import InputPreprocessor, OutputPreprocessor
import pickle
import tensorflow as tf
from silx.io.dictdump import h5todict
DEBUG = False

if DEBUG:
    N_TRAIN = 1280
    N_TEST = 512
    LOAD_PATH = "data/debug"

else:
    N_TRAIN = 1024000
    N_TEST = 16384
    LOAD_PATH = "data"
noise_scales = [np.round(x, 2) for x in np.arange(0, 1.0, 1.0/20)]


def save_training_data_and_stats(n_samples_train):
    q_values = DefaultTrainedModel().q_values
    sample = DefaultTrainedModel().sample
    generator = ReflectivityGenerator(q_values, sample)
    labels = generator.generate_random_labels(n_samples_train)
    reflectivity = generator.simulate_reflectivity(labels)
    labels = labels.drop(labels.columns.difference(
        ["Film_thickness", "Film_roughness", "Film_sld"]), axis=1)

    mean_labels, std_labels = np.mean(labels, axis=0), np.std(labels, axis=0)
    np.savetxt(f"{LOAD_PATH}/labels_real_scale_forvisual.csv", labels)
    labels = (labels-mean_labels)/std_labels


    np.savetxt(f"{LOAD_PATH}/reflectivity_real_scale.csv", reflectivity)
    np.savetxt(f"{LOAD_PATH}/labels_unit_scale.csv", labels)

    np.savetxt(f"{LOAD_PATH}/mean_labels.csv", mean_labels)
    np.savetxt(f"{LOAD_PATH}/std_labels.csv", std_labels)
    for noise_scale in noise_scales:
        uniform_noise_range_low = 1-noise_scale
        uniform_noise_range_high = 1+noise_scale
        reflectivity_noise = reflectivity*tf.random.uniform(
            shape=reflectivity.shape, minval=uniform_noise_range_low, maxval=uniform_noise_range_high)
        mean_data, std_data = np.mean(reflectivity_noise, axis=0), np.std(
            reflectivity_noise, axis=0)

        np.savetxt(f"{LOAD_PATH}/mean_data_{noise_scale}.csv", mean_data)
        np.savetxt(f"{LOAD_PATH}/std_data_{noise_scale}.csv", std_data)


def save_test_data_and_stats(n_samples_test):
    q_values = DefaultTrainedModel().q_values
    sample = DefaultTrainedModel().sample
    generator = ReflectivityGenerator(q_values, sample)
    labels = generator.generate_random_labels(n_samples_test)
    ip = InputPreprocessor()
    out = OutputPreprocessor(sample)
    reflectivity_real_scale = generator.simulate_reflectivity(labels)

    labels_full = labels.copy()
    labels = labels.drop(labels.columns.difference(
        ["Film_thickness", "Film_roughness", "Film_sld"]), axis=1)
    labels = labels.to_numpy()
    for noise_scale in noise_scales:
        uniform_noise_range_low = 1-noise_scale
        uniform_noise_range_high = 1+noise_scale
        reflectivity_real_scale = reflectivity_real_scale*np.random.uniform(
            uniform_noise_range_low, uniform_noise_range_high, reflectivity_real_scale.shape)
        np.savetxt(
            f"{LOAD_PATH}/test_reflectivity_real_scale_{noise_scale}.csv", reflectivity_real_scale)

    mean_labels, std_labels = np.loadtxt(
        f"{LOAD_PATH}/mean_labels.csv"), np.loadtxt(f"{LOAD_PATH}/std_labels.csv")
    labels = (labels-mean_labels)/std_labels
    np.savetxt(f"{LOAD_PATH}/test_labels_unit_scale.csv", labels)
    with open(f"{LOAD_PATH}/test_data_objects", 'wb') as outp:
        pickle.dump([generator, ip, out, q_values, labels_full],
                    outp, pickle.HIGHEST_PROTOCOL)

def iterate_experiments():
    test_refl_lst = []
    test_q_values_lst = []
    lables_lst = []
    pen2 = h5todict("evaluation/xrr_dataset.h5")
    for experiment_name in pen2.keys():
        test_refl = pen2[experiment_name]["experiment"]["data"]
        test_q_values = pen2[experiment_name]["experiment"]["q"]
        lables = [pen2[experiment_name]["fit"]["Film_thickness"],
                  pen2[experiment_name]["fit"]["Film_roughness"],
                  np.real(pen2[experiment_name]["fit"]["Film_sld"])]
        test_refl_lst.append(test_refl)
        test_q_values_lst.append(test_q_values)
        lables_lst.append(lables)
    return test_refl_lst, test_q_values_lst, lables_lst


def main():
    save_training_data_and_stats(n_samples_train=N_TRAIN)
    save_test_data_and_stats(n_samples_test=N_TEST)
if __name__ == "__main__":
    main()