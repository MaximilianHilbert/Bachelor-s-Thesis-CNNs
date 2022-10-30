import numpy as np
import tensorflow as tf

from lib import helper

PARAMETERS = ["Film_thickness", "Film_roughness", "Film_sld"]


def generate_analysis_pipeline(data_dict, model_object):
    mse_dict = get_mse(data_dict, model_object)
    log_error_dict = get_log_error_exp(data_dict)
    absolute_error_array = get_absolute_error(data_dict)
    relative_error_array = get_relative_error(absolute_error_array, data_dict)
    return mse_dict, log_error_dict, absolute_error_array, relative_error_array


def log_error(y_pred, y):
    """calculated the log error

    Args:
        y_pred (np.array): array containing the predicted refl values
        y (np.array): array containing the ground truth refl values

    Returns:
        float: value of the calculated log error per refl curve
    """
    return np.mean(np.absolute((np.log10(y_pred) - np.log10(y))), axis=1)


def get_mse(data_dict, model_obj):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    error_dict = {}
    for modelname in ["param_sim_mlp", "param_sim_cnn"]:
        simulated_parameters_true_scale = data_dict[modelname][PARAMETERS].to_numpy()
        true_labels_real_scale = data_dict["true_labels"][PARAMETERS].to_numpy()
        simulated_parameters_unit_scale = helper.normalize(
            simulated_parameters_true_scale, model_obj.mean_labels, model_obj.std_labels
        )
        true_parameters_unit_scale = helper.normalize(
            true_labels_real_scale, model_obj.mean_labels, model_obj.std_labels
        )
        error_dict[modelname[-3:]] = mse(
            simulated_parameters_unit_scale, true_parameters_unit_scale
        ).numpy()
    return error_dict


def get_log_error(data_dict):
    return {
        modelname[-3:]: log_error(
            data_dict[modelname], data_dict["refl_true_real_scale"]
        )
        for modelname in ["sim_mlp", "sim_cnn"]
    }


def get_log_error_exp(data_dict):
    return_dict = {}
    for model_name in ["sim_mlp", "sim_cnn"]:
        final_log_errors = []
        list_of_lists = [
            log_error(real, sim)
            for real, sim in zip(
                data_dict[model_name], data_dict["refl_true_real_scale"]
            )
        ]
        for experiment in list_of_lists:
            final_log_errors.extend(iter(experiment))
        return_dict[model_name[-3:]] = final_log_errors
    return return_dict


def get_absolute_error(data_dict):
    return {
        model_id[-3:]: np.absolute(
            data_dict[model_id][PARAMETERS].to_numpy()
            - data_dict["true_labels"][PARAMETERS].to_numpy()
        )
        for model_id in ["param_sim_mlp", "param_sim_cnn"]
    }


def get_relative_error(abs_error_dict, data_dict):
    return {
        modelname: (
            100 * abs_error_dict[modelname] / data_dict["true_labels"][PARAMETERS]
        ).to_numpy()
        for modelname in ["mlp", "cnn"]
    }

