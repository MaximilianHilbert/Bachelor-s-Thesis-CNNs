import tensorflow as tf
import datetime
from tensorboard.plugins.hparams import api as hp
import os
from lib import pipeline
from lib import data_gen
import sys
import numpy as np
import pickle
import pandas as pd
import subprocess
from silx.io.dictdump import h5todict
from mlreflect.models import DefaultTrainedModel, TrainedModel
N_REFL = 109
def_model = DefaultTrainedModel()
q_values_used_for_training = def_model.q_values
sample = def_model.sample
DEBUG = False
LR=0.001
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if DEBUG:
    noise_level, num_filter_1, num_filter_2, num_filter_3, kernel_size_1, kernel_size_2, kernel_size_3, batch_size=0.2, 25, 50, 75, 12, 8, 4, 256
    num_units=(np.floor((np.floor((np.floor((N_REFL-kernel_size_1+1)/2)-kernel_size_2+1)/2)-kernel_size_3+1)/2))*num_filter_3
    N_EPOCH = 2
    DEBUG_PATH="debug/"


else:
    noise_level = float(sys.argv[1])
    num_filter_1 = int(float(sys.argv[2]))
    num_filter_2 = int(float(sys.argv[3]))
    num_filter_3 = int(float(sys.argv[4]))
    kernel_size_1 = int(float(sys.argv[5]))
    kernel_size_2 = int(float(sys.argv[6]))
    kernel_size_3 = int(float(sys.argv[7]))
    batch_size = int(float(sys.argv[8]))
    num_units=(np.floor((np.floor((np.floor((N_REFL-kernel_size_1+1)/2)-kernel_size_2+1)/2)-kernel_size_3+1)/2))*num_filter_3
    N_EPOCH = 150
    DEBUG_PATH=""


SAVE_STRING = f'{noise_level}_{num_units}_{num_filter_1}_{num_filter_2}_{num_filter_3}_{kernel_size_1}_{kernel_size_2}_{kernel_size_3}_{batch_size}_{current_time}'
logs = os.path.join("logs_new_metric/", DEBUG_PATH, f"{SAVE_STRING}")
print("Hyperparameters used")
print(SAVE_STRING)

# Hyperparameter Training Configuration
HP_NUM_FILTER_1 = hp.HParam("num_filter_1", hp.Discrete([num_filter_1]))
HP_NUM_FILTER_2 = hp.HParam("num_filter_2", hp.Discrete([num_filter_2]))
HP_NUM_FILTER_3 = hp.HParam("num_filter_3", hp.Discrete([num_filter_3]))
HP_KERNEL_SIZE_1 = hp.HParam("kernel_size_1", hp.Discrete([kernel_size_1]))
HP_KERNEL_SIZE_2 = hp.HParam("kernel_size_2", hp.Discrete([kernel_size_2]))
HP_KERNEL_SIZE_3 = hp.HParam("kernel_size_3", hp.Discrete([kernel_size_3]))
HP_NOISE_LEVEL = hp.HParam("noise_level", hp.Discrete([noise_level]))
HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([batch_size]))


#Load Data from disc
with open(os.path.join("data/", DEBUG_PATH, "test_data_objects"), "rb") as inp:
    generator, ip, out, q_values, labels_full = pickle.load(inp)

data_train_real_scale, labels_train_unit_scale = np.loadtxt(
    os.path.join("data/", DEBUG_PATH, "reflectivity_real_scale.csv")), np.loadtxt(os.path.join("data/", DEBUG_PATH, "labels_unit_scale.csv"))
data_test_real_scale, labels_test_unit_scale = np.loadtxt(
   os.path.join("data/", DEBUG_PATH, f"test_reflectivity_real_scale_{noise_level}.csv")), np.loadtxt(os.path.join("data/", DEBUG_PATH, "test_labels_unit_scale.csv"))

mean_data, std_data, mean_labels, std_labels = np.loadtxt(f"data/{DEBUG_PATH}mean_data_{noise_level}.csv"), np.loadtxt(
   os.path.join("data/", DEBUG_PATH, f"std_data_{noise_level}.csv")), np.loadtxt(os.path.join("data/", DEBUG_PATH, "mean_labels.csv")), np.loadtxt(os.path.join("data/", DEBUG_PATH, "std_labels.csv"))

n_samples_test=len(data_test_real_scale)
#Convert Data to Tensors for handling within noise layer
mean_data = tf.convert_to_tensor(mean_data, dtype=np.float32)
std_data = tf.convert_to_tensor(std_data, dtype=np.float32)
mean = tf.reshape(mean_data, [N_REFL, 1])
std = tf.reshape(std_data, [N_REFL, 1])

combined_dict = {"data_test": data_test_real_scale, "labels_test": labels_test_unit_scale,
                 "data_train": data_train_real_scale, "labels_train": labels_train_unit_scale}
#Generate efficient Datasets out of train/test/validation data
(train_dataset_real_scale, test_dataset_real_scale), validation_dataset_unit_scale = pipeline.create_train_val_test_datasets(
    combined_dict, batch_size)

#Define model and noise layer
def get_model(hparams):
    def get_config(self):
        cfg = super().get_config()
        return cfg

    class noise_and_standardize(tf.keras.layers.Layer):
        def get_config(self):
            cfg = super().get_config()
            return cfg

        def __init__(self, noise_level):
            super(noise_and_standardize, self).__init__()
            self.noise_level = noise_level

        def call(self, input, training):
            if training:
                uniform_noise_range_low = 1-noise_level
                uniform_noise_range_high = 1+noise_level
                noisy = input*tf.random.uniform(
                    shape=input.shape, minval=uniform_noise_range_low, maxval=uniform_noise_range_high)
                output = tf.divide(tf.subtract(noisy, mean), std)
            else:
                output = tf.divide(tf.subtract(input, mean), std)
            return output

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(N_REFL, 1), batch_size=batch_size),
        noise_and_standardize(hparams[HP_NOISE_LEVEL]),
        tf.keras.layers.Conv1D(
            hparams[HP_NUM_FILTER_1], kernel_size=hparams[HP_KERNEL_SIZE_1], padding="valid", activation="relu"),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(
            hparams[HP_NUM_FILTER_2], kernel_size=hparams[HP_KERNEL_SIZE_2], padding="valid", activation="relu"),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(
            hparams[HP_NUM_FILTER_3], kernel_size=hparams[HP_KERNEL_SIZE_3], padding="valid", activation="relu"),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_units, activation='relu'),
        tf.keras.layers.Dense(3)
    ])
    return model


def testing_loss(model):
    #Test on synthetic data
    absolute_error_array_synth, mse_errors_synth = pipeline.test_on_syn_data_in_pipeline(
        n_samples_test, test_dataset_real_scale, labels_test_unit_scale, model, N_REFL, mean_labels, std_labels)
    
    #save results for synthetic data for further analysis
    np.savetxt(
        os.path.join("evaluation_errors/", DEBUG_PATH, f"{SAVE_STRING}_mse_errors_synth.csv"), mse_errors_synth)
    np.savetxt(
        os.path.join("evaluation_errors/", DEBUG_PATH, f"{SAVE_STRING}_absolute_errors_synth.csv"), absolute_error_array_synth)

    #Test on experimental data and save results to disc
    test_refl_lst, test_q_values_lst, lables_lst = data_gen.iterate_experiments()
    th_lst, rh_lst, sld_lst, param_error_lst = pipeline.test_on_exp_data_pipeline(
        test_refl_lst, test_q_values_lst, lables_lst, q_values_used_for_training, "CNN", sample, model, noise_level, mean_labels, std_labels, mean_data, std_data)
    
    np.savetxt(
        os.path.join("evaluation_errors/", DEBUG_PATH, f"{SAVE_STRING}_mse_errors_exp.csv"), param_error_lst)
    np.savetxt(os.path.join("evaluation_errors/", DEBUG_PATH, f"{SAVE_STRING}_th.csv"), th_lst)
    np.savetxt(os.path.join("evaluation_errors/", DEBUG_PATH, f"{SAVE_STRING}_rh.csv"), rh_lst)
    np.savetxt(os.path.join("evaluation_errors/", DEBUG_PATH, f"{SAVE_STRING}_sld.csv"), sld_lst)

    return mse_errors_synth, absolute_error_array_synth
#Main Training and logging loop
def main():
    hparams = {HP_NOISE_LEVEL: noise_level,
            HP_NUM_FILTER_1: num_filter_1, HP_NUM_FILTER_2: num_filter_2, HP_NUM_FILTER_3: num_filter_3,
            HP_KERNEL_SIZE_1: kernel_size_1, HP_KERNEL_SIZE_2: kernel_size_2, HP_KERNEL_SIZE_3: kernel_size_3, HP_BATCH_SIZE: batch_size}

    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logs,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)

    model = get_model(hparams)

    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    model.compile(loss="mse",
                  optimizer=opt,
                  metrics=["mae", "mape"],
                  )

    model.fit(train_dataset_real_scale,
              epochs=N_EPOCH,
              callbacks=[hp.KerasCallback(
                  logs, hparams), early_stopping_cb, tb_cb],
              verbose=2,
              validation_data=validation_dataset_unit_scale,
              )
    print(model.summary())
    mse_errors, absolute_error_array_synth=testing_loss(model)
    mse_median=np.median(mse_errors)
    param_errors_median=np.median(absolute_error_array_synth, axis=1)
    model.save(os.path.join("models/", DEBUG_PATH, f"{SAVE_STRING}"))
    
    #file_writer = tf.summary.create_file_writer(f'logs/hp/{SAVE_STRING}_log_loss')

    # with file_writer.as_default():
    #     tf.summary.scalar("Fin_log_loss", data=test_loss, step=1)

    # if mse_median>0.15 or param_errors_median[0]>20 or param_errors_median[1]>5 or param_errors_median[2]>0.2:
    #     joined_model_path=os.path.join("models/", DEBUG_PATH, f"{SAVE_STRING}")
    #     joined_log_path=os.path.join("logs_new_metric/", DEBUG_PATH, f"{SAVE_STRING}")
    #     subprocess.call(f"rm -r {joined_model_path}", shell=True)
    #     subprocess.call(f"rm -r {joined_log_path}", shell=True)

    # subprocess.call(f"rm -r logs/hp/{SAVE_STRING}_log_loss", shell=True)
    # subprocess.call(f"rm -r logs/hp/{SAVE_STRING}", shell=True)
    # subprocess.call(f"rm -r models/{SAVE_STRING}", shell=True)
    # os.remove(f"evaluation/log_errors_{SAVE_STRING}.csv")
if __name__ == "__main__":
    main()
