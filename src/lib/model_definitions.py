import numpy as np
import tensorflow as tf


def get_model(hp_config_object, data_loader_object):
    num_units = (
        np.floor(
            (
                np.floor(
                    (
                        np.floor(
                            (
                                data_loader_object.number_of_points_per_curve
                                - 2 * hp_config_object.kernel_sizes[0]
                                + 2
                            )
                            / 2
                        )
                        - 2 * hp_config_object.kernel_sizes[1]
                        + 2
                    )
                    / 2
                )
                - 2 * hp_config_object.kernel_sizes[2]
                + 2
            )
            / 2
        )
        * hp_config_object.number_of_filters[2]
        * 0.5
    )

    class NoiseAndStandardize(tf.keras.layers.Layer):
        def get_config(self):
            return super().get_config()

        def __init__(self, noise_level):
            super(NoiseAndStandardize, self).__init__()
            self.noise_level = noise_level

        def call(self, input_batch, training):
            if not training:
                return tf.divide(
                    tf.subtract(input_batch, data_loader_object.mean_data),
                    data_loader_object.std_data,
                )

            uniform_noise_range_low = 1 - hp_config_object.noise_level
            uniform_noise_range_high = 1 + hp_config_object.noise_level
            noisy = input_batch * tf.random.uniform(
                shape=input_batch.shape,
                minval=uniform_noise_range_low,
                maxval=uniform_noise_range_high,
            )
            return tf.divide(
                tf.subtract(noisy, data_loader_object.mean_data),
                data_loader_object.std_data,
            )

    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(
                shape=(data_loader_object.number_of_points_per_curve, 1),
                batch_size=hp_config_object.batch_size,
            ),
            NoiseAndStandardize(hp_config_object.noise_level),
            tf.keras.layers.Conv1D(
                hp_config_object.number_of_filters[0],
                kernel_size=hp_config_object.kernel_sizes[0],
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.Conv1D(
                hp_config_object.number_of_filters[0],
                kernel_size=hp_config_object.kernel_sizes[0],
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Conv1D(
                hp_config_object.number_of_filters[1],
                kernel_size=hp_config_object.kernel_sizes[1],
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.Conv1D(
                hp_config_object.number_of_filters[1],
                kernel_size=hp_config_object.kernel_sizes[1],
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Conv1D(
                hp_config_object.number_of_filters[2],
                kernel_size=hp_config_object.kernel_sizes[2],
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.Conv1D(
                hp_config_object.number_of_filters[2],
                kernel_size=hp_config_object.kernel_sizes[2],
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_units, activation="relu"),
            tf.keras.layers.Dense(3),
        ]
    )
