import matplotlib.pyplot as plt
import numpy as np

from lib import global_


def gen_mse_plots(mse_dict):
    fig_boxplot_mse, axs_boxplot_mse = plt.subplots(1)
    axs_boxplot_mse.set_ylabel("MSE")
    axs_boxplot_mse.boxplot(
        [mse_dict[modelname] for modelname in ["mlp", "cnn"]], showfliers=False
    )
    axs_boxplot_mse.set_xticklabels(["MLP", "CNN"])
    fig_boxplot_mse.tight_layout()


def gen_log_plots(log_dict):
    fig_boxplot_log_error, axs_boxplot_log_error = plt.subplots(1)
    axs_boxplot_log_error.boxplot(
        [log_dict[modelname] for modelname in ["mlp", "cnn"]], showfliers=False
    )
    axs_boxplot_log_error.set_xticklabels(["MLP", "CNN"])
    axs_boxplot_log_error.set_ylabel("Log-error")
    fig_boxplot_log_error.tight_layout()


def gen_random_curve_indices(data_dict, sim, exp_id):
    if sim:
        indices = np.arange(data_dict["refl_true_real_scale"].shape[0])
    else:
        indices = np.arange(data_dict["test_q_values_lst"][exp_id].shape[0])
    np.random.shuffle(indices)
    return indices


def gen_curves_plots(data_dict, sim=True):
    fig_curves, axs_curves = plt.subplots(
        3, 3, figsize=(15, 15), sharex=True, sharey=True
    )
    q_values = global_.Q_VALUES if sim else data_dict["test_q_values_lst"][0]
    for name in ["sim_mlp", "sim_cnn"]:
        for axis, curve_pred in zip(
            axs_curves.flat, data_dict[name] if sim else data_dict[name][0]
        ):
            if not sim:
                curve_pred = np.atleast_2d(curve_pred)[0, :]
            axis.semilogy(
                q_values,
                curve_pred,
                label=f"prediction {name[-3:]}",
                color="firebrick" if name[-3:] == "cnn" else "gray",
            )
    if sim:
        refl_ground_truth = data_dict["refl_true_real_scale"]
    else:
        refl_ground_truth = data_dict["refl_true_real_scale"][0]
    for axis, ground_truth_curve in zip(axs_curves.flat, refl_ground_truth):
        axis.semilogy(
            q_values, ground_truth_curve, label="ground truth", color=global_.CORAL,
        )

    for axis in axs_curves[-1, :]:
        axis.set_xlabel(r"q in $\mathrm{\AA^{-1}}$")
    for axis in axs_curves[:, 0]:
        axis.set_ylabel("Reflectivity")
    for axis in axs_curves.flat:
        axis.legend(loc=1)
    fig_curves.tight_layout()


def gen_param_plots(absolute_dict, relative_dict):
    fig_boxplots_param_errors, axs_boxplots_param_errors = plt.subplots(
        2, 3, figsize=(15, 10)
    )
    for idx, box in enumerate(axs_boxplots_param_errors[0, :]):
        box.boxplot([absolute_dict[modelname][:, idx] for modelname in ["mlp", "cnn"]])
        if idx == 0:
            box.set_title("Film Thickness")
        if idx == 1:
            box.set_title("Film Roughness")
        if idx == 2:
            box.set_title("Film SLD")

        if idx != 2:
            box.set_ylabel(r"Absolute error in $\mathrm{\AA}$")
        else:
            box.set_ylabel(r"Absolute error in $\mathrm{10^{-6} \AA^{-2}}$")
        box.set_xticklabels(["MLP", "CNN"])

    for idx, box in enumerate(axs_boxplots_param_errors[1, :]):
        box.boxplot([relative_dict[modelname][:, idx] for modelname in ["mlp", "cnn"]])
        box.set_xticklabels(["MLP", "CNN"])
        box.set_ylabel("Relative error in %")
    fig_boxplots_param_errors.tight_layout()
