import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from mlreflect.data_generation import interp_reflectivity, ReflectivityGenerator
from mlreflect.training import InputPreprocessor, OutputPreprocessor
from mlreflect.models import DefaultTrainedModel, TrainedModel
from mlreflect.curve_fitter import CurveFitter
from tensorflow import keras
PARAMETERS = ["Film_thickness", "Film_roughness", "Film_sld"]


def create_train_val_test_datasets(combined_dict, BATCH_SIZE):
    train_data = combined_dict["data_train"]
    train_labels = combined_dict["labels_train"]
    test_data = combined_dict["data_test"]
    test_labels = combined_dict["labels_test"]
    train_number_of_curves = len(train_data)
    train_values_per_curve = len(train_data[0, :])
    # create datasets of train and test
    train_data = train_data.reshape(
        train_number_of_curves, train_values_per_curve, 1)
    #train_data=train_data.reshape(None, train_values_per_curve)
    x_train, x_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=0.2)
    datasets_real_scale = []
    datasets_real_scale.append(tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(BATCH_SIZE))  # .cache()

    test_number_of_curves = len(test_data)
    test_values_per_curve = len(test_data[0, :])

    test_data = test_data.reshape(
        test_number_of_curves, test_values_per_curve, 1)
    datasets_real_scale.append(tf.data.Dataset.from_tensor_slices(
        (test_data, test_labels)).batch(BATCH_SIZE))  # .cache()
    # create dataset of validation data
    validation_dataset_real_scale = tf.data.Dataset.from_tensor_slices(
        (x_val, y_val)).batch(BATCH_SIZE)  # .prefetch(tf.data.experimental.AUTOTUNE)
    return datasets_real_scale, validation_dataset_real_scale


def evaluation(n_samples_test, test_dataset_real_scale, model, N_REFL):
    model.build((n_samples_test, N_REFL, 1))
    pred_unit_scale = model.predict(test_dataset_real_scale)
    return pred_unit_scale


def log_error(y_pred, y):
    return np.nanmean(np.absolute(np.log10(y_pred)-np.log10(y)))


def interp_predict_mlp_test(test_refl, test_q_values):
    trained_model_mlp = DefaultTrainedModel()
    curve_fitter_mlp = CurveFitter(trained_model_mlp)
    fit_results_mlp = curve_fitter_mlp.fit_curve(
        test_refl, test_q_values, polish=False, optimize_q=False)
    return fit_results_mlp


def interp_predict_cnn_test(test_refl, q_values_used_for_training, test_q_values, sample, keras_model_cnn, noise_level, mean_labels, std_labels, mean_data, std_data):
    

    class CurveFitterCNN:
        def __init__(self, trained_model: TrainedModel):
            self.trained_model = trained_model

            self.generator = None

            self.ip = InputPreprocessor()
            self.ip._standard_std = trained_model.ip_std
            self.ip._standard_mean = trained_model.ip_mean

            self.op = OutputPreprocessor(trained_model.sample, 'min_to_zero')

        def _interpolate_intensity(self, intensity, q_values):
            intensity = np.atleast_2d(intensity)
            interp_intensity = np.empty((len(intensity), len(self.trained_model.q_values)))
            for i in range(len(intensity)):
                interp_intensity[i] = interp_reflectivity(self.trained_model.q_values, q_values, intensity[i])
            return interp_intensity

        def fit_curve(self, corrected_curve, q_values, dq: float = 0, factor: float = 1, polish=False,
                      fraction_bounds: tuple = (0.5, 0.5, 0.1), optimize_q=True, n_q_samples: int = 1000,
                      optimize_scaling=False, n_scale_samples: int = 300, simulate_reflectivity=True) -> dict:

            max_q_idx = abs(
                q_values - self.trained_model.q_values.max()).argmin() + 1
            min_q_idx = abs(
                q_values - self.trained_model.q_values.min()).argmin()

            corrected_curve = np.atleast_2d(corrected_curve)
            interpolated_curve = self._interpolate_intensity(
                corrected_curve * factor, q_values + dq)
            generator = ReflectivityGenerator(
                q_values, self.trained_model.sample)

            n_curves = len(corrected_curve)
            
            if not (optimize_scaling & optimize_q):
                ###########################################################################################################################################################changes#######################################
                # interpolated_curve=(interpolated_curve-mean_data)/std_data
                predicted_parameters = self.trained_model.keras_model.predict(
                interpolated_curve.reshape(interpolated_curve.shape[0], interpolated_curve.shape[1], 1))
                predicted_parameters = predicted_parameters*std_labels+mean_labels

                dummy_params = self.op.restore_labels(predicted_parameters)
                dummy_params["Film_thickness"] = predicted_parameters[:, 0]
                dummy_params["Film_roughness"] = predicted_parameters[:, 1]
                dummy_params["Film_sld"] = predicted_parameters[:, 2]
                self._ensure_positive_parameters(dummy_params)

            if simulate_reflectivity:
                predicted_refl = generator.simulate_reflectivity(dummy_params, progress_bar=False)
            else:
                predicted_refl = None
            return {'predicted_reflectivity': predicted_refl, 'predicted_parameters': dummy_params}

        @staticmethod
        def _ensure_positive_parameters(parameters):
                for parameter_name in parameters.columns:
                    if 'thickness' in parameter_name or 'roughness' in parameter_name:
                        parameters[parameter_name] = abs(parameters[parameter_name])

    trained_model_cnn = TrainedModel()
    q_values_used_for_training = DefaultTrainedModel().q_values
    sample = DefaultTrainedModel().sample

    trained_model_cnn.from_variable(model=keras_model_cnn,
                                    sample=sample,
                                    q_values=q_values_used_for_training,  # ask
                                    ip_mean=mean_data,
                                    ip_std=std_data)
    curve_fitter_cnn = CurveFitterCNN(trained_model_cnn)
    pred_refl_cnn = curve_fitter_cnn.fit_curve(
        test_refl, test_q_values, polish=False, optimize_q=False)
    return pred_refl_cnn


def pred_and_calculate_errors(test_refl_lst, test_q_values_lst, test_lables_lst, q_values_used_for_training, model_name, sample, model_cnn, noise_level, mean_labels, std_labels, mean_data, std_data):
    th_lst = []
    rh_lst = []
    sld_lst = []
    refl_sim_lst = []
    pred_lables_lst = []
    log_error_lst = []
    # loop over all test datasets available
    for curve_idx, (test_curves, q_values) in enumerate(zip(test_refl_lst, test_q_values_lst)):
        if model_name == "MLP":
            pred_dict = interp_predict_mlp_test(
                test_curves, q_values)
        else:
            pred_dict = interp_predict_cnn_test(
                test_curves, q_values_used_for_training, q_values, sample, model_cnn, noise_level, mean_labels, std_labels, mean_data, std_data)
        pred_lables_lst.append(pred_dict["predicted_parameters"][[
            "Film_thickness", "Film_roughness", "Film_sld"]])
        refl_sim_lst.append(pred_dict["predicted_reflectivity"])

    # first model that gets served is MLP
    for lables_set, pred_set in zip(test_lables_lst, pred_lables_lst):
        pred_set = pred_set.to_numpy()
        for param_idx, param in enumerate(PARAMETERS):
            error_lst = []
            # special treatment for single values
            if pred_set[:, param_idx].shape != lables_set[param_idx].shape:
                error_lst.append(np.absolute(
                    lables_set[param_idx]-pred_set[:, param_idx])[0])
            # base case of multiple values in list
            else:
                # Kontrolle
                for absolute_errors in np.absolute(lables_set[param_idx]-pred_set[:, param_idx]):
                    error_lst.append(absolute_errors)
            for element in error_lst:
                if param == "Film_thickness":
                    th_lst.append(element)
                if param == "Film_roughness":
                    rh_lst.append(element)
                if param == "Film_sld":
                    sld_lst.append(element)
    for curve_number, (curve_true, curve_sim) in enumerate(zip(test_refl_lst, refl_sim_lst)):
        log_error_lst.append(log_error(curve_true, curve_sim))
    th_mean = np.nanmedian(th_lst)
    rh_mean = np.nanmedian(rh_lst)
    sld_mean = np.nanmedian(sld_lst)
    log_mean = np.nanmedian(log_error_lst)
    print(f"{model_name}: ")
    print(f"Exp Median param errors: {th_mean} {rh_mean} {sld_mean}")
    print(f"Exp Median log error: {log_mean}")
    return th_lst, rh_lst, sld_lst, refl_sim_lst, log_error_lst


def test_on_syn_data_in_pipeline(SAVE_STRING, n_samples_test, test_dataset_real_scale, data_test_real_scale, labels_test_unit_scale, model, N_REFL, mean_labels, std_labels, lables_full, gen_test):
    pred_params_unit_scale = evaluation(
        n_samples_test, test_dataset_real_scale, model, N_REFL)
    pred_params_real_scale = pred_params_unit_scale*std_labels+mean_labels
    pred_params_real_scale = pd.DataFrame(pred_params_real_scale, columns=[
                                          "Film_thickness", "Film_roughness", "Film_sld"])

    # copy labels df to get additional label content
    lables_full["Film_thickness"] = pred_params_real_scale.to_numpy()[:, 0]
    lables_full["Film_roughness"] = pred_params_real_scale.to_numpy()[:, 1]
    lables_full["Film_sld"] = pred_params_real_scale.to_numpy()[:, 2]
    R_pred_real_scale = gen_test.simulate_reflectivity(lables_full)

    log_error_arr = np.zeros_like(R_pred_real_scale)
    for idx, (curve_true, curve_pred) in enumerate(zip(data_test_real_scale, R_pred_real_scale)):
        error_per_curve = log_error(curve_true, curve_pred)
        log_error_arr[idx] = error_per_curve

    params_test_real_scale = labels_test_unit_scale*std_labels+mean_labels

    absolute_error_array = np.absolute(
        pred_params_real_scale.to_numpy()-params_test_real_scale)
    param_median = np.nanmedian(absolute_error_array, axis=0)
    log_error_median = np.nanmedian(log_error_arr)
    print(f"Synth median param error: {param_median}")
    print(f"Synth log error median: {log_error_median}")
    return absolute_error_array, log_error_arr
