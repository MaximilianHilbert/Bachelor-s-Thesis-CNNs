import datetime
import os
import sys

import tensorflow as tf

from lib import config, model_definitions, pipeline

DEBUG = True
LEARNING_RATE = 0.001
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if DEBUG:
    hp_config_object = config.HyperparameterConfig(
        system_parameters=[0.1, 45, 20, 15, 4, 2, 3, 256]
    )
    N_EPOCH = 2
    DEBUG = "debug"


else:
    hp_config_object = config.HyperparameterConfig(system_parameters=sys.argv[1:])
    N_EPOCH = 150
    DEBUG = ""


save_string = f"{hp_config_object.noise_level}_{hp_config_object.number_of_filters[0]}_{hp_config_object.number_of_filters[1]}_{hp_config_object.number_of_filters[2]}_{hp_config_object.kernel_sizes[0]}_{hp_config_object.kernel_sizes[1]}_{hp_config_object.kernel_sizes[2]}_{hp_config_object.batch_size}_{current_time}"
logs = os.path.join("logs_new_metric3l", DEBUG, f"{save_string}")

print("Hyperparameters used")
print(save_string)

data_loader_object = config.DataLoader(
    debug_path=DEBUG, hp_config_object=hp_config_object
)


data_loader_object.set_train_val_test_datasets(hp_config_object=hp_config_object)

model = model_definitions.get_model(hp_config_object, data_loader_object)


def testing_loss(trained_model):
    mse_median = pipeline.test_on_syn_data_in_pipeline(
        dataloader=data_loader_object, trained_model_instance=trained_model
    )
    if not DEBUG:
        savepath = os.path.join(
            "evaluation_errors3l", DEBUG, f"{save_string}_mse_value_synth.csv"
        )
        with open(savepath, "w") as file:
            file.write(str(mse_median))


def main():

    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logs,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        loss="mse", optimizer=opt, metrics=["mae", "mape"],
    )

    model.fit(
        data_loader_object.train_dataset_real_scale,
        epochs=N_EPOCH,
        callbacks=[tb_cb, early_stopping_cb],
        verbose=2,
        validation_data=data_loader_object.validation_dataset_real_scale,
    )
    print(model.summary())
    testing_loss(model)
    model.save(os.path.join("models3l", DEBUG, f"{save_string}"))


if __name__ == "__main__":
    main()
