import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from lib import helper


class HyperparameterConfig:
    def __init__(self, system_parameters):
        self.noise_level = float(system_parameters[0])
        self.number_of_filters = list(map(int, system_parameters[1:4]))
        self.kernel_sizes = list(map(int, system_parameters[4:7]))
        self.batch_size = int(system_parameters[7])


class ModelLoader:
    def __init__(self, model_path, noise_level):
        self.model = keras.models.load_model(model_path)
        self.mean_labels = np.loadtxt("data/mean_labels.csv")
        self.std_labels = np.loadtxt("data/std_labels.csv")
        self.mean_data = np.loadtxt(f"data/mean_data_{noise_level}.csv")
        self.std_data = np.loadtxt(f"data/std_data_{noise_level}.csv")


class DataLoader:
    def __init__(self, debug_path, hp_config_object):
        self.data_train_real_scale = np.loadtxt(
            os.path.join("data", debug_path, "reflectivity_real_scale.csv")
        )
        self.labels_train_unit_scale = np.loadtxt(
            os.path.join("data", debug_path, "labels_unit_scale.csv")
        )

        self.data_test_real_scale = np.loadtxt(
            os.path.join(
                "data",
                debug_path,
                f"test_reflectivity_real_scale_{hp_config_object.noise_level}.csv",
            )
        )
        self.labels_test_unit_scale = np.loadtxt(
            os.path.join("data", debug_path, "test_labels_unit_scale.csv")
        )

        self.mean_data = tf.reshape(
            tf.convert_to_tensor(
                np.loadtxt(
                    os.path.join(
                        "data",
                        debug_path,
                        f"mean_data_{hp_config_object.noise_level}.csv",
                    )
                ),
                dtype=np.float32,
            ),
            [109, 1],
        )
        self.std_data = tf.reshape(
            tf.convert_to_tensor(
                np.loadtxt(
                    os.path.join(
                        "data",
                        debug_path,
                        f"std_data_{hp_config_object.noise_level}.csv",
                    )
                ),
                dtype=np.float32,
            ),
            [109, 1],
        )

        self.mean_labels = np.loadtxt(
            os.path.join("data", debug_path, "mean_labels.csv")
        )
        self.std_labels = np.loadtxt(os.path.join("data", debug_path, "std_labels.csv"))

        self.number_of_points_per_curve = 109

        self.train_dataset_real_scale = None
        self.test_dataset_real_scale = None
        self.validation_dataset_real_scale = None

        self.pred_params_unit_scale = None
        self.pred_params_real_scale = None
        self.params_test_real_scale = None

    def set_train_val_test_datasets(self, hp_config_object):
        train_data = helper.reshape_into_3d(self.data_train_real_scale)
        test_data = helper.reshape_into_3d(self.data_test_real_scale)

        x_train, x_val, y_train, y_val = train_test_split(
            train_data, self.labels_train_unit_scale, test_size=0.2
        )
        train_dataset_real_scale = helper.create_dataset_from_tensor_slices(
            x_slice=x_train, y_slice=y_train, batch_size=hp_config_object.batch_size
        )
        test_dataset_real_scale = helper.create_dataset_from_tensor_slices(
            x_slice=test_data,
            y_slice=self.labels_test_unit_scale,
            batch_size=hp_config_object.batch_size,
        )

        x_val_noisy = x_val * np.random.uniform(
            low=1 - hp_config_object.noise_level,
            high=1 + hp_config_object.noise_level,
            size=x_val.shape,
        )
        validation_dataset_real_scale = tf.data.Dataset.from_tensor_slices(
            (x_val_noisy, y_val)
        ).batch(hp_config_object.batch_size)

        self.train_dataset_real_scale = train_dataset_real_scale
        self.test_dataset_real_scale = test_dataset_real_scale
        self.validation_dataset_real_scale = validation_dataset_real_scale

    def evaluation(self, model):
        model.build(
            (len(self.data_test_real_scale), self.number_of_points_per_curve, 1)
        )
        self.pred_params_unit_scale = model.predict(self.test_dataset_real_scale)

    def rescale_unit_to_real(self):
        self.pred_params_real_scale = (
            self.pred_params_unit_scale * self.std_labels + self.mean_labels
        )
        self.params_test_real_scale = (
            self.labels_test_unit_scale * self.std_labels + self.mean_labels
        )

