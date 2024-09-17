import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
# from Dataset import Dataset


class HyperParameterTuner:
    def __init__(self, data):
        self.data = data

    def dnn_model_builder(
        self,
        hp,
        hp_units_min: int = 32,
        hp_units_max: int = 512,
        hp_units_step: int = 32,
        hp_layers_min: int = 1,
        hp_layers_max: int = 5,
        hp_layers_step: int = 1,
        hp_learning_rates: [float] = [1e-1, 1e-2, 1e-3, 1e-4],
        hp_loss: str = "mae",
    ) -> tf.keras.Model:
        hp_units = hp.Int(
            "units", min_value=hp_units_min, max_value=hp_units_max, step=hp_units_step
        )
        hp_layers = hp.Int(
            "layers",
            min_value=hp_layers_min,
            max_value=hp_layers_max,
            step=hp_layers_step,
        )
        hp_learning_rate = hp.Choice("learning_rate", values=hp_learning_rates)

        model = keras.Sequential()
        model.add(keras.layers.Flatten())
        
        for _ in range(hp_layers):  # Add hp_layers dense layers
            model.add(keras.layers.Dense(units=hp_units, activation="relu"))

        # model.add(keras.layers.Dense(units=hp_units, activation="relu"))
        # model.add(hp_layers)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=hp_loss,
            metrics=["accuracy"],
        )
        return model

    def kt_dnn_tuner(
        self,
        epochs: int = 1000,
        factor: int = 5,
        directory: str = "dnn",
        hyperband_interations: int = 3,
    ) -> []:
        tuner = kt.Hyperband(
            self.dnn_model_builder,
            objective="val_accuracy",
            max_epochs=int(epochs / 5),
            factor=factor,
            directory=f"{self.data.folder_name}/dnn/",
            project_name=f"Cool Project",
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25)
        tuner.search(
            self.data.train_data,
            self.data.train_labels,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[stop_early],
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        return best_hps