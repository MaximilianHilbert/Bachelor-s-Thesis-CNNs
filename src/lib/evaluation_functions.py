import numpy as np
import pandas as pd
import tensorflow as tf
from mlreflect.data_generation import ReflectivityGenerator, interp_reflectivity
from mlreflect.models import DefaultTrainedModel, TrainedModel
from mlreflect.training import InputPreprocessor, OutputPreprocessor

from lib import data_gen, global_, pipeline_mlp


def predict_mlp_syn(gen_test, ip, out, data_test_true_real_scale):
    reflectivity_true_unit_scale = ip.standardize(data_test_true_real_scale)
    model = DefaultTrainedModel().keras_model
    pred_params_unit_scale = model.predict(reflectivity_true_unit_scale)
    pred_params_real_scale_full = out.restore_labels(pred_params_unit_scale)
    reflectivity_sim = gen_test.simulate_reflectivity(pred_params_real_scale_full)
    return reflectivity_sim, pred_params_real_scale_full


def simulate_and_predict(q_values, number_of_curves, batch_size, model_object):
    sample = DefaultTrainedModel().sample
    generator = ReflectivityGenerator(q_values, sample)
    true_labels = generator.generate_random_labels(number_of_curves)
    ip = InputPreprocessor()
    out = OutputPreprocessor(sample)
    reflectivity_true_real_scale = generator.simulate_reflectivity(true_labels)
    reflectivity_sim_mlp, params_mlp = predict_mlp_syn(
        generator, ip, out, reflectivity_true_real_scale
    )

    reflectivity_sim_cnn, params_cnn = predict_cnn_syn(
        generator, model_object, reflectivity_true_real_scale, true_labels, batch_size
    )

    return {
        "sim_mlp": reflectivity_sim_mlp,
        "sim_cnn": reflectivity_sim_cnn,
        "param_sim_mlp": params_mlp,
        "param_sim_cnn": params_cnn,
        "refl_true_real_scale": reflectivity_true_real_scale,
        "true_labels": true_labels,
    }


def predict_cnn_syn(
    generator, model_object, reflectivity_true_real_scale, labels, batch_size
):

    reflectivity_true_real_scale = reflectivity_true_real_scale.reshape(-1, 109, 1)
    pred_params_true_scale_full = labels.copy()

    labels = labels.drop(
        labels.columns.difference(["Film_thickness", "Film_roughness", "Film_sld"]),
        axis=1,
    )
    labels = labels.to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (reflectivity_true_real_scale, labels)
    ).batch(batch_size)
    pred_params_unit_scale = model_object.model.predict(dataset)
    pred_params_true_scale = (
        pred_params_unit_scale * model_object.std_labels + model_object.mean_labels
    )

    pred_params_true_scale_full["Film_thickness"] = pred_params_true_scale[:, 0]
    pred_params_true_scale_full["Film_roughness"] = pred_params_true_scale[:, 1]
    pred_params_true_scale_full["Film_sld"] = pred_params_true_scale[:, 2]
    reflectivity_sim = generator.simulate_reflectivity(pred_params_true_scale_full)

    return reflectivity_sim, pred_params_true_scale_full


def organize_experimental_data():
    test_refl_lst, test_q_values_lst, test_lables_lst = data_gen.iterate_experiments()
    param_clean_test_list = []
    for experiment in test_lables_lst:
        thickness_lst, roughness_lst, sld_lst = experiment
        param_clean_test_list.extend(
            list(
                zip(
                    np.atleast_1d(thickness_lst),
                    np.atleast_1d(roughness_lst),
                    np.atleast_1d(sld_lst),
                )
            )
        )
    return test_refl_lst, test_q_values_lst, np.array(param_clean_test_list)


def make_experimental_prediction(
    test_refl_lst, test_q_values_lst, model_object, param_unit_scale_true
):
    return_dict = {}
    for model_name in ["mlp", "cnn"]:
        pred_lables_lst, refl_sim_lst = [], []
        for test_curves, q_values in zip(test_refl_lst, test_q_values_lst):
            if model_name == "mlp":
                pred_dict = pipeline_mlp.interp_predict_mlp_test(test_curves, q_values)
            else:
                pred_dict = interp_predict_cnn_test(test_curves, q_values, model_object)
            pred_lables_lst.append(
                pred_dict["predicted_parameters"][
                    ["Film_thickness", "Film_roughness", "Film_sld"]
                ]
            )
            refl_sim_lst.append(pred_dict["predicted_reflectivity"])
        return_dict[f"param_sim_{model_name}"] = pd.concat(
            pred_lables_lst, ignore_index=True
        )
        return_dict[f"sim_{model_name}"] = np.array(refl_sim_lst)
        return_dict["true_labels"] = pd.DataFrame(
            param_unit_scale_true, columns=global_.PARAMETERS
        )
    return_dict["refl_true_real_scale"] = np.array(test_refl_lst)
    return_dict["test_q_values_lst"] = np.array(test_q_values_lst)
    return return_dict


def interp_predict_cnn_test(test_refl, test_q_values, model_object):
    """adapted from mlreflect
    """

    class CurveFitterCNN:
        def __init__(self, trained_model: TrainedModel):
            self.trained_model = trained_model

            self.generator = None

            self.ip = InputPreprocessor()
            self.ip._standard_std = trained_model.ip_std
            self.ip._standard_mean = trained_model.ip_mean

            self.op = OutputPreprocessor(trained_model.sample, "min_to_zero")

        def _interpolate_intensity(self, intensity, q_values):
            intensity = np.atleast_2d(intensity)
            interp_intensity = np.empty(
                (len(intensity), len(self.trained_model.q_values))
            )
            for i in range(len(intensity)):
                interp_intensity[i] = interp_reflectivity(
                    self.trained_model.q_values, q_values, intensity[i]
                )
            return interp_intensity

        def fit_curve(
            self,
            corrected_curve,
            q_values,
            dq: float = 0,
            factor: float = 1,
            polish=False,
            fraction_bounds: tuple = (0.5, 0.5, 0.1),
            optimize_q=True,
            n_q_samples: int = 1000,
            optimize_scaling=False,
            n_scale_samples: int = 300,
            simulate_reflectivity=True,
        ) -> dict:

            max_q_idx = abs(q_values - self.trained_model.q_values.max()).argmin() + 1
            min_q_idx = abs(q_values - self.trained_model.q_values.min()).argmin()

            corrected_curve = np.atleast_2d(corrected_curve)
            interpolated_curve = self._interpolate_intensity(
                corrected_curve * factor, q_values + dq
            )
            generator = ReflectivityGenerator(q_values, self.trained_model.sample)

            n_curves = len(corrected_curve)

            if not (optimize_scaling & optimize_q):
                predicted_parameters = self.trained_model.keras_model.predict(
                    interpolated_curve.reshape(
                        interpolated_curve.shape[0], interpolated_curve.shape[1], 1
                    )
                )
                predicted_parameters = (
                    predicted_parameters * model_object.std_labels
                    + model_object.mean_labels
                )

                dummy_params = self.op.restore_labels(predicted_parameters)
                dummy_params["Film_thickness"] = predicted_parameters[:, 0]
                dummy_params["Film_roughness"] = predicted_parameters[:, 1]
                dummy_params["Film_sld"] = predicted_parameters[:, 2]
                self._ensure_positive_parameters(dummy_params)

            if simulate_reflectivity:
                predicted_refl = generator.simulate_reflectivity(
                    dummy_params, progress_bar=False
                )
            else:
                predicted_refl = None
            return {
                "predicted_reflectivity": predicted_refl,
                "predicted_parameters": dummy_params,
            }

        @staticmethod
        def _ensure_positive_parameters(parameters):
            for parameter_name in parameters.columns:
                if "thickness" in parameter_name or "roughness" in parameter_name:
                    parameters[parameter_name] = abs(parameters[parameter_name])

    trained_model_cnn = TrainedModel()

    trained_model_cnn.from_variable(
        model=model_object.model,
        sample=global_.SAMPLE,
        q_values=global_.Q_VALUES,
        ip_mean=model_object.mean_data,
        ip_std=model_object.std_data,
    )
    curve_fitter_cnn = CurveFitterCNN(trained_model_cnn)
    return curve_fitter_cnn.fit_curve(
        test_refl, test_q_values, polish=False, optimize_q=False
    )

