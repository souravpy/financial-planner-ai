import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from CNNLSTM import CNNLSTM
from Stocks import Stocks
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

class HyperParameterTuner:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def dnn_model_builder(
        self,
        hp,
        hp_units_min: int = 32,
        hp_units_max: int = 128,
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

        model.add(keras.layers.Dense(units=hp_units, activation="relu"))
        model.add(keras.layers.Dense(hp_layers))

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
        hyperband_interations: int = 1,
    ) -> []:
        tuner = kt.Hyperband(
            self.dnn_model_builder,
            objective="val_accuracy",
            max_epochs=int(epochs / 5),
            factor=factor,
            directory=f"results/dnn/",
            project_name=f"all_stocks_dnn",
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25)
        tuner.search(
            self.train_data,
            self.train_labels,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[stop_early],
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(
            f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}. The best number of layers if {best_hps.get('layers')}.
        """
        )

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            self.train_data,
            self.train_labels,
            validation_split=0.2,
            verbose=0,
            epochs=500,
        )

        val_acc_per_epoch = history.history["val_accuracy"]
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print("Best epoch: %d" % (best_epoch,))

        hist = pd.DataFrame(history.history)
        hist["epoch"] = history.epoch
        print(hist.tail(20))

        CNNLSTM.plot(history)
        CNNLSTM.summarize_model(model)
        CNNLSTM.evaluate_model(model, self.test_data, self.test_labels)
        CNNLSTM.preditcted_plot(model, self.test_data, self.test_labels)
        test_results = model.evaluate(self.test_data, self.test_labels, verbose=0)

        return best_hps

    def random_forest(self):
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.train_data, self.train_labels)
        predictions = rf.predict(self.test_data)
        errors = abs(predictions - self.test_labels)
        mape = 100 * (errors / self.test_labels)
        accuracy = 100 - np.mean(mape)
        print("Accuracy:", round(accuracy, 2), "%.")
        print(
            pd.Series(
                rf.feature_importances_, index=self.train_data.columns
            ).sort_values(ascending=False)
        )
        return rf
    
    def remove_nan(self):
        self.train_data.dropna(inplace=True)
        self.train_labels.dropna(inplace=True)
        self.test_data.dropna(inplace=True)
        self.test_labels.dropna(inplace=True)


if __name__ == "__main__":
    stocks = Stocks("../stock_market_data")
    # stocks.combine_data()

    # stocks.data_pipeline()
    # stocks.load_data("stocks_train.csv", "stocks_val.csv", "stocks_test.csv")
    stocks.load_data("AAPL_train.csv", "AAPL_val.csv", "AAPL_test.csv")
    train_x = stocks.train_df
    train_y = train_x.pop("Close")
    val_x = stocks.val_df
    val_y = val_x.pop("Close")
    test_x = stocks.test_df
    test_y = test_x.pop("Close")
    tuner = HyperParameterTuner(train_x, train_y, test_x, test_y)
    tuner.remove_nan()
    # tuner.kt_dnn_tuner()
    tuner.random_forest()
