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
mse=tf.keras.losses.MeanSquaredError()

def create_train_val_test_datasets(combined_dict, BATCH_SIZE, noise_level):
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
    uniform_noise_range_low = 1-noise_level
    uniform_noise_range_high = 1+noise_level
    x_val = x_val*np.random.uniform(low=uniform_noise_range_low, high=uniform_noise_range_high, size=x_val.shape)
    validation_dataset_real_scale = tf.data.Dataset.from_tensor_slices(
        (x_val, y_val)).batch(BATCH_SIZE)  # .prefetch(tf.data.experimental.AUTOTUNE)
    return datasets_real_scale, validation_dataset_real_scale


def evaluation(n_samples_test, test_dataset_real_scale, model, N_REFL):
    model.build((n_samples_test, N_REFL, 1))
    pred_unit_scale = model.predict(test_dataset_real_scale)
    return pred_unit_scale


def log_error(y_pred, y):
    return np.mean(np.absolute((np.log10(y_pred)-np.log10(y))))

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

def calculate_absolute_error_lists(pred_lables_lst, test_lables_lst):
    th_lst, rh_lst, sld_lst, error_lst=[], [], [], []
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
                    for absolute_errors in np.absolute(lables_set[param_idx]-pred_set[:, param_idx]):
                        error_lst.append(absolute_errors)
                for element in error_lst:
                    if param == "Film_thickness":
                        th_lst.append(element)
                    if param == "Film_roughness":
                        rh_lst.append(element)
                    if param == "Film_sld":
                        sld_lst.append(element)

    return th_lst, rh_lst, sld_lst
    
def test_on_exp_data_pipeline(test_refl_lst, test_q_values_lst, test_lables_lst, q_values_used_for_training, model_name, sample, model_cnn, noise_level, mean_labels, std_labels, mean_data, std_data):
    refl_sim_lst=[]
    pred_lables_lst = []
    param_error_lst=[]
    log_error_lst=[]
    th_lst, rh_lst, sld_lst=[], [], []
    mse=tf.keras.losses.MeanSquaredError()
    # loop over all test datasets available
    for curve_idx, (test_curves, q_values) in enumerate(zip(test_refl_lst, test_q_values_lst)):
        if model_name == "MLP":
            pred_dict = interp_predict_mlp_test(
                test_curves, q_values)
        else:
            pred_dict =interp_predict_cnn_test(
                test_curves, q_values_used_for_training, q_values, sample, model_cnn, noise_level, mean_labels, std_labels, mean_data, std_data)
        pred_lables_lst.append(pred_dict["predicted_parameters"][[
            "Film_thickness", "Film_roughness", "Film_sld"]])
        refl_sim_lst.append(pred_dict["predicted_reflectivity"])
    param_clean_test_list=[]
    # norm. params and save to list
    for experiment in test_lables_lst:
        for param_set in zip(np.atleast_1d(experiment[0]).tolist(), np.atleast_1d(experiment[1]).tolist(), np.atleast_1d(experiment[2]).tolist()):
            param_set=(param_set-mean_labels)/std_labels
            param_clean_test_list.append(param_set)
    # mse error and norm. pred values
    for pred_set in pred_lables_lst:
        pred_set=(pred_set-mean_labels)/std_labels
        for curve_param_true, curve_param_pred in zip(np.atleast_1d(param_clean_test_list), np.atleast_1d(pred_set.to_numpy())):
            param_error_lst.append(mse(curve_param_true, curve_param_pred).numpy())

    for curve_number, (curve_true, curve_sim) in enumerate(zip(test_refl_lst, refl_sim_lst)):
        log_error_lst.append(log_error(curve_true, curve_sim))        
    th_lst, rh_lst, sld_lst=calculate_absolute_error_lists(pred_lables_lst, test_lables_lst)

    print(f"{model_name}")
    print(f"Exp median param error: {np.median(th_lst), np.median(rh_lst), np.median(sld_lst)}")
    print(f"Exp mse median: {np.median(param_error_lst)}")
    print(f"Exp log error median: {np.median(np.array(log_error_lst))}")
    return th_lst, rh_lst, sld_lst, param_error_lst, log_error_lst


def test_on_syn_data_in_pipeline(n_samples_test, test_dataset_real_scale, labels_test_unit_scale, model, N_REFL, mean_labels, std_labels):
    pred_params_unit_scale = evaluation(
        n_samples_test, test_dataset_real_scale, model, N_REFL)
    pred_params_real_scale = pred_params_unit_scale*std_labels+mean_labels
    pred_params_real_scale = pd.DataFrame(pred_params_real_scale, columns=[
                                          "Film_thickness", "Film_roughness", "Film_sld"])

    mse_lst=[]
    params_test_real_scale = labels_test_unit_scale*std_labels+mean_labels
    for param_set_true, param_set_pred in zip(labels_test_unit_scale, pred_params_unit_scale):
        mse_lst.append(mse(param_set_true, param_set_pred).numpy())
    absolute_error_array = np.absolute(
        pred_params_real_scale.to_numpy()-params_test_real_scale)
    param_median = np.median(absolute_error_array, axis=0)
    mse_median = np.median(mse_lst)
    print(f"Synth median param error: {param_median}")
    print(f"Synth mse median: {mse_median}")
    return absolute_error_array, mse_lst

if __name__=="__main__":
    print()