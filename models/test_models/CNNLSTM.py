import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    TimeDistributed,
)
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.utils import plot_model

from sklearn.metrics import (
    explained_variance_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
)
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

import matplotlib.pyplot as plt

from Stocks import Stocks
from Visualizations import DataWindow


class CNNLSTM:
    def build_model() -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(
            TimeDistributed(
                Conv1D(64, kernel_size=3, activation="relu", input_shape=(None, 100, 1))
            )
        )
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation="relu")))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation="relu")))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Flatten()))
        # model.add(Dense(5, kernel_regularizer=L2(0.01)))

        # LSTM layers
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(100, return_sequences=False)))
        model.add(Dropout(0.5))

        # Final layers
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
        return model

    def train(model: tf.keras.Sequential, train_x, train_y, val_x, val_y):
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        history = model.fit(
            train_x,
            train_y,
            epochs=32,
            validation_data=(val_x, val_y),
            callbacks=[early_stopping],
            verbose=1,
            shuffle=True,
        )
        return history

    def plot(history):
        plt.plot(history.history["loss"], label="train loss")
        plt.plot(history.history["val_loss"], label="val loss")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig("results/train_loss.png")

        plt.plot(history.history["mse"], label="train mse")
        plt.plot(history.history["val_mse"], label="val mse")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig("results/train_mse.png")

        plt.plot(history.history["mae"], label="train mae")
        plt.plot(history.history["val_mae"], label="val mae")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig("results/train_mae.png")

    def summarize_model(model):
        print(model.summary())
        plot_model(
            model,
            to_file="results/model_plot.png",
            show_shapes=True,
            show_layer_names=True,
        )
        model.save("results/model.h5")

    def evaluate_model(model, test_x, test_y):
        loss, mse, mae = model.evaluate(test_x, test_y)
        print("loss: ", loss)
        print("mse: ", mse)
        print("mae: ", mae)
        df = pd.DataFrame(
            [[loss, mse, mae]],
            columns=["loss", "mse", "mae"],
        )
        df.to_csv("results/evaluation.csv", index=False)

    def stats(model, test_x, test_y):
        y_pred = model.predict(test_x, verbose=0)
        y_pred = y_pred[:, 0]
        print("explained_variance_score: ", explained_variance_score(test_y, y_pred))
        print("mean_poisson_deviance: ", mean_poisson_deviance(test_y, y_pred))
        print("mean_gamma_deviance: ", mean_gamma_deviance(test_y, y_pred))
        print("r2_score: ", r2_score(test_y, y_pred))
        print("max_error: ", max_error(test_y, y_pred))
        df = pd.DataFrame(
            [
                [
                    explained_variance_score(test_y, y_pred),
                    mean_poisson_deviance(test_y, y_pred),
                    mean_gamma_deviance(test_y, y_pred),
                    r2_score(test_y, y_pred),
                    max_error(test_y, y_pred),
                ]
            ],
            columns=[
                "explained_variance_score",
                "mean_poisson_deviance",
                "mean_gamma_deviance",
                "r2_score",
                "max_error",
            ],
        )
        df.to_csv("results/stats.csv", index=False)

    def preditcted_plot(model, test_x, test_y):
        y_pred = model.predict(test_x, verbose=0)
        y_true = test_y
        plt.plot(y_pred, label="Predicted")
        plt.plot(y_true, label="Real")
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()
        plt.savefig("results/predicted.png")


if __name__ == "__main__":
    stocks = Stocks("../stock_market_data")
    # stocks.combine_data()

    # stocks.data_pipeline()
    stocks.load_data("stocks_train.csv", "stocks_val.csv", "stocks_test.csv")
    train_x = stocks.train_df
    train_y = train_x.pop("Close")
    val_x = stocks.val_df
    val_y = val_x.pop("Close")
    test_x = stocks.test_df
    test_y = test_x.pop("Close")
    model = CNNLSTM.build_model()
    history = CNNLSTM.train(model, train_x, train_y, val_x, val_y)
    CNNLSTM.plot(history)
    CNNLSTM.summarize_model(model)
    CNNLSTM.evaluate_model(model, test_x, test_y)
    CNNLSTM.stats(model, test_x, test_y)
    CNNLSTM.preditcted_plot(model, test_x, test_y)
