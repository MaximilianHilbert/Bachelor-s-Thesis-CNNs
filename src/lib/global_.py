import matplotlib
import tensorflow as tf
from mlreflect.models import DefaultTrainedModel

SAMPLE = DefaultTrainedModel().sample
Q_VALUES = DefaultTrainedModel().q_values
MODEL_MLP = DefaultTrainedModel()
KERAS_MLP = DefaultTrainedModel().keras_model

PARAMETERS = ["Film_thickness", "Film_roughness", "Film_sld"]
PARAMETERS_PLAIN = ["Thickness", "Roughness", "SLD"]
SILVER = "#BFC0C0"
CORAL = "#4F5D75"

mse = tf.keras.losses.MeanSquaredError()
matplotlib.rcParams.update(
    {
        "font.size": 20,
        "boxplot.boxprops.color": "gray",
        "boxplot.patchartist": True,
        "boxplot.showfliers": False,
        "boxplot.notch": False,
        "boxplot.medianprops.color": "firebrick",
        "patch.facecolor": SILVER,
    }
)
