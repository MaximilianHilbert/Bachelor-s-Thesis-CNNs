import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lib import pipeline
from mlreflect.models import DefaultTrainedModel, TrainedModel
import pickle
from mlreflect.data_generation import ReflectivityGenerator
import tensorflow as tf
from tensorflow import keras
test_number_of_curves=100
test_values_per_curve=109

fig, axs=plt.subplots(5,5,figsize=(15,10), sharex=True)#, sharey=True)
fig_boxplot, axs_boxplot=plt.subplots(1,2, sharey=True)
fig_boxplots_param, axs_boxplots_param=plt.subplots(1,1, sharey=True)
fig_boxplots_param.suptitle("Parameter comparison")
fig.suptitle("Ground truth vs. Predicted curves (simulations, no noise, pseudo test)")
fig_boxplot.suptitle("Statistical comparison")
n_samples=100
q_values = DefaultTrainedModel().q_values
sample = DefaultTrainedModel().sample
generator = ReflectivityGenerator(q_values, sample)
labels = generator.generate_random_labels(n_samples)
reflectivity = generator.simulate_reflectivity(labels)
#uniform_noise_range = (0.7, 1.3)

labels=labels.drop(labels.columns.difference(["Film_thickness", "Film_roughness", "Film_sld"]), axis=1)
labels=labels.to_numpy()
mean_data_test, std_data_test=np.mean(reflectivity, axis=0), np.std(reflectivity, axis=0)
mean_labels_test, std_labels_test=np.mean(labels, axis=0), np.std(labels, axis=0)
labels_test=(labels-mean_labels_test)/std_labels_test
reflectivity_true=(reflectivity-mean_data_test)/std_data_test

for idx_model, modelname in enumerate(["mlp","cnn"]):
    error_lst=[]
    # with open(f'./evaluation/test_data', 'rb') as inp:
    #     generator, (reflectivity_true, lables_test, mean_data, std_data, mean_labels_test, std_labels_test) = pickle.load(inp)
    reflectivity_true_q_scale=reflectivity_true*std_data_test+mean_data_test

    if modelname=="mlp":
        reflectivity_true=reflectivity_true.reshape(test_number_of_curves, test_values_per_curve)
        model=DefaultTrainedModel().keras_model
        pred_params_unit_scale=model.predict(reflectivity_true) #different scaling while training?
    else:
        model=keras.models.load_model('models/0.2_60_15_30_16_8_20220609-104729')
        reflectivity_true=reflectivity_true.reshape(test_number_of_curves, test_values_per_curve, 1)
        #test_dataset_unit_scale=tf.data.Dataset.from_tensor_slices((reflectivity_true, lables_test)).batch(64)
        pred_params_unit_scale=model.predict(reflectivity_true)
    pred_params_q_scale=pred_params_unit_scale*std_labels_test+mean_labels_test
    pred_params_q_scale=pd.DataFrame(pred_params_q_scale, columns=["thickness", "roughness", "sld"])

    generator=ReflectivityGenerator(q_values, sample)
    reflectivity_sim = generator.simulate_reflectivity(pred_params_q_scale)
    bool=np.isfinite(reflectivity_sim).all()
    np.testing.assert_array_almost_equal(reflectivity, reflectivity_true_q_scale,decimal=6, err_msg='', verbose=True)
    
    for idx_lineplot, ax in enumerate(axs.flat):
        ax.semilogy(reflectivity_sim[idx_lineplot], label=f"predicted_{modelname}")
        ax.legend()
    for curve_true, curve_sim in zip(reflectivity_true, reflectivity_sim):
        curve_true=curve_true.reshape(test_values_per_curve)
        error_lst.append(pipeline.log_error(curve_true, curve_sim))
    arr=np.array(error_lst)
    axs_boxplot[idx_model].boxplot(arr)
    axs_boxplot[idx_model].set_title(f"{modelname}")

for idx, ax in enumerate(axs.flat):
    ax.semilogy(reflectivity_true_q_scale[idx], label="ground truth")
    ax.legend()
fig.tight_layout()
plt.show()

# fig_boxplot.tight_layout()

for idx, ax in enumerate(axs.flat):
    ax.semilogy(reflectivity_true_q_scale[idx], label="ground truth")
    ax.legend()
fig.tight_layout()
plt.show()