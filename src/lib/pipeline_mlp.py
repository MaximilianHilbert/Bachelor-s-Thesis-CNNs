import numpy as np
from mlreflect.curve_fitter import CurveFitter
from mlreflect.models import DefaultTrainedModel


def evaluation(n_samples_test, test_dataset_real_scale, model, n_refl):
    model.build((n_samples_test, n_refl, 1))
    return model.predict(test_dataset_real_scale)


def log_error(y_pred, y_true):
    return np.mean(np.absolute((np.log10(y_pred) - np.log10(y_true))))


def interp_predict_mlp_test(test_refl, test_q_values):
    trained_model_mlp = DefaultTrainedModel()
    curve_fitter_mlp = CurveFitter(trained_model_mlp)
    return curve_fitter_mlp.fit_curve(
        test_refl, test_q_values, polish=False, optimize_q=False
    )
