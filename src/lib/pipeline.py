import numpy as np
import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()


def test_on_syn_data_in_pipeline(dataloader, trained_model_instance):
    dataloader.evaluation(trained_model_instance)
    dataloader.rescale_unit_to_real()

    mse_array = np.array(
        [
            mse(param_set_true, param_set_pred).numpy()
            for param_set_true, param_set_pred in zip(
                dataloader.labels_test_unit_scale, dataloader.pred_params_unit_scale
            )
        ]
    )

    absolute_error_array = np.absolute(
        dataloader.pred_params_real_scale - dataloader.params_test_real_scale
    )
    absolute_median_error_array = np.median(absolute_error_array, axis=0)
    mse_median = np.median(mse_array)

    print(
        f"Synth median param errors (thickness, roughness, sld): {absolute_median_error_array}"
    )
    print(f"Synth mse median: {mse_median}")

    return mse_median
