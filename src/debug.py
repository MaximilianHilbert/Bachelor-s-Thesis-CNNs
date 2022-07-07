from lib import pipeline
from cgi import test
from operator import index
from statistics import mean
import tensorflow as tf
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
from tqdm import tqdm
from current_Training import N_REFL
from lib import data_gen
import numpy as np
import matplotlib.pyplot as plt
import os
def log_error(y_pred, y):
    return np.mean(np.absolute(np.log(y_pred)-np.log(y)))

master_dict=pipeline.build_pipeline(BATCH_SIZE=64, ROOT_DIR="/home/maximilian/Dropbox/Studium/Bachelorarbeit/src",VAL_PERC=0.2)
test_data_unit_scale=master_dict["test_d_us"]
test_targets_unit_scale=master_dict["test_t_us"]
test_data_q_scale=master_dict["test_d_qs"]
test_targets_q_scale=master_dict["test_t_qs"]
mean_data=master_dict["mean_d"]
mean_labels=master_dict["mean_l"]
std_data=master_dict["std_d"]
std_labels=master_dict["std_l"]
fig, axs=plt.subplots(5,5,figsize=(15,10), sharex=True, sharey=True)
fig.suptitle("Ground truth vs. Predicted curves (simulations, pseudo test)")
for dir in os.listdir("/home/maximilian/Dropbox/Studium/Bachelorarbeit/src/models"):
    model=tf.keras.models.load_model("/home/maximilian/Dropbox/Studium/Bachelorarbeit/src/models/"+str(dir))
    pred_lst_unit_scale=pipeline.predict_values(test_data_unit_scale, model)
    
    #change scale
    pred_lst_q_scale=(pred_lst_unit_scale*std_labels)+mean_labels
    for idx, ax in enumerate(axs.flat):
        d=pred_lst_q_scale[idx][0]
        r=pred_lst_q_scale[idx][1]
        sld=pred_lst_q_scale[idx][2]
        params={
        "D": d,
        "R": r,
        "SLD": sld,
        "SUBSTRATE_R": 10,
        "SUBSTRATE_SLD": 16e-6              #noch fixiert
        }
        try:
            R, _=data_gen.get_curve(2, params)
        except ValueError:
            print("There was a negative Value in Data!")
        log_loss=np.round(log_error(R[0], test_data_q_scale[idx]), 4)

        ax.semilogy(R[0], label=f"pred nlv: {dir[:3]} E: {log_loss}")
for idx, ax in enumerate(axs.flat):
    ax.semilogy(test_data_q_scale[idx], label="ground truth")
    ax.legend()
#plt.tight_layout()
plt.show()