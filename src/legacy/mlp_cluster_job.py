from queue import SimpleQueue
from mlreflect.models import DefaultTrainedModel, TrainedModel
from mlreflect.data_generation import ReflectivityGenerator, noise
from mlreflect.training import InputPreprocessor, OutputPreprocessor, NoiseGenerator
from mlreflect.curve_fitter import CurveFitter
from mlreflect.utils.naming import make_timestamp
from tensorflow import keras
import os
import numpy as np
from lib import pipeline_mlp

batch_size=256
noise_level=0.3

keras_model=DefaultTrainedModel()._keras_model
print(keras_model.summary())
data_train_real_scale, labels_train_unit_scale = np.loadtxt(
    os.path.join("data/", "reflectivity_real_scale.csv")), np.loadtxt(os.path.join("data/", "labels_unit_scale.csv"))

combined_dict = {"data_test": data_train_real_scale, "labels_test": labels_train_unit_scale,
                 "data_train": data_train_real_scale, "labels_train": labels_train_unit_scale}
(train_dataset_real_scale, test_dataset_real_scale), validation_dataset_unit_scale = pipeline_mlp.create_train_val_test_datasets(
    combined_dict, batch_size, noise_level)

output = keras_model.fit(train_dataset_real_scale, validation_data=validation_dataset_unit_scale,
                   epochs=20, verbose=1)