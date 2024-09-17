import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.preprocessing import MinMaxScaler

from Visualizations import DataWindow

import datetime
import os

from tensorflow.keras import Model, Sequential

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell

import warnings

warnings.filterwarnings("ignore")


class Stocks:
    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.markets = ["nasdaq", "nyse", "sp500"]
        self.train_per = 0.7
        self.val_per = 0.2
        self.test_per = 0.1
        self.column_indices = None
        self.failed = []
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_data(self, train: str, val: str, test: str) -> None:
        self.train_df = pd.DataFrame(pd.read_csv(train))
        self.val_df = pd.DataFrame(pd.read_csv(val))
        self.test_df = pd.DataFrame(pd.read_csv(test))
        self.column_indices = self.get_column_indices(self.train_df)

    def save(self, df: pd.DataFrame, file_name: str) -> None:
        df.to_csv(file_name, index=False, header=True)

    def save_failed(self) -> None:
        df = pd.DataFrame(self.failed)
        df.to_csv("failed.csv", index=False, header=True)

    def convert_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["Day"] = df["Date"].dt.day
        df = df.drop(["Date"], axis=1)
        return df

    def check_dataset_size(self, df: pd.DataFrame) -> bool:
        return len(df) >= 100

    def get_column_indices(self, df: pd.DataFrame) -> dict:
        return {name: i for i, name in enumerate(df.columns)}

    def split_dataset(
        self, df: pd.DataFrame
    ) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        length = len(df)
        train = df[0 : int(length * self.train_per)]
        val = df[
            int(length * self.train_per) : int(length * (self.train_per + self.val_per))
        ]
        test = df[int(length * (self.train_per + self.val_per)) :]

        return train, val, test

    def normalize(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scaler = MinMaxScaler()
        scaler.fit(train)
        train[train.columns] = scaler.transform(train[train.columns])
        val[val.columns] = scaler.transform(val[val.columns])
        test[test.columns] = scaler.transform(test[test.columns])
        return train, val, test

    def combine_data(self) -> None:
        train = pd.DataFrame()
        val = pd.DataFrame()
        test = pd.DataFrame()
        for market in self.markets:
            try:
                file_list = os.listdir(f"{self.folder_name}/{market}/csv/")
                for file in file_list:
                    if file.endswith("_train.csv"):
                        temp_train = pd.DataFrame(
                            pd.read_csv(f"{self.folder_name}/{market}/csv/{file}")
                        )
                        train = pd.concat([train, temp_train])
                    elif file.endswith("_val.csv"):
                        temp_val = pd.DataFrame(
                            pd.read_csv(f"{self.folder_name}/{market}/csv/{file}")
                        )
                        val = pd.concat([val, temp_val])
                    elif file.endswith("_test.csv"):
                        temp_test = pd.DataFrame(
                            pd.read_csv(f"{self.folder_name}/{market}/csv/{file}")
                        )
                        test = pd.concat([test, temp_test])
            except Exception as e:
                print(e)
                continue

        self.save(train, "stocks_train.csv")
        self.save(val, "stocks_val.csv")
        self.save(test, "stocks_test.csv")

    def data_pipeline(self):
        for market in self.markets:
            file_list = os.listdir(f"{self.folder_name}/{market}/csv/")
            for file in file_list:
                try:
                    print(f"Processing market: {market} and file: {file}")
                    if (
                        file.endswith("_train.csv")
                        or file.endswith("_val.csv")
                        or file.endswith("_test.csv")
                    ):
                        continue
                    df = pd.DataFrame(
                        pd.read_csv(f"{self.folder_name}/{market}/csv/{file}")
                    )
                    if not self.check_dataset_size(df):
                        self.failed.append(f"{market}/csv/{file}")
                        continue

                    df = self.convert_date(df)
                    train, val, test = self.split_dataset(df)
                    train_norm, val_norm, test_norm = self.normalize(train, val, test)
                    self.save(
                        train_norm,
                        f"{self.folder_name}/{market}/csv/{file[:-4]}_train.csv",
                    )
                    self.save(
                        val_norm,
                        f"{self.folder_name}/{market}/csv/{file[:-4]}_val.csv",
                    )
                    self.save(
                        test_norm,
                        f"{self.folder_name}/{market}/csv/{file[:-4]}_test.csv",
                    )
                except Exception as e:
                    print(f"Failed to process: {market}/csv/{file}")
                    print(e)
                    self.failed.append(f"{market}/csv/{file}")
                    continue

        self.column_indices = self.get_column_indices(train)
        self.save_failed()
        self.combine_data()

    def compile_and_fit(self, model, window, patience=10, max_epochs=50):
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, mode="min"
        )
        model.compile(
            loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteError()]
        )

        history = model.fit(
            window.train,
            epochs=max_epochs,
            validation_data=window.val,
            callbacks=[early_stopping],
        )
        return history

    def fit(self, model, window, patience=10, max_epochs=50):
        model.fit(
            window.train,
            epochs=max_epochs,
            validation_data=window.val,
            callbacks=[early_stopping],
        )

    def train(self, linear, train_df, val_df, test_df):
        multi_window = DataWindow(
            input_width=21,
            label_width=21,
            shift=21,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["Close"],
        )
        self.fit(linear, multi_window)


class MultiStepLastBaseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return tf.tile(inputs[:, -1:, :], [1, 21, 1])
        return tf.tile(inputs[:, -1:, self.label_index :], [1, 21, 1])


class RepeatBaseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        return inputs[:, :, self.label_index :]


class AutoRegressive(Model):
    def __init__(self, units, out_steps, train_df, val_df, test_df):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = LSTMCell(units)
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        self.dense = Dense(train_df.shape[1])

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)

        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)

            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


if __name__ == "__main__":
    folder_name = "../../../stock_market_data/sp500/csv/"
    stocks = Stocks("../../../stock_market_data/sp500/csv/")
    # stocks.combine_data()

    # stocks.data_pipeline()
    # stocks.load_data("stocks_train.csv", "stocks_val.csv", "stocks_test.csv")
    stocks.load_data(
        f"{folder_name}IBM_train.csv",
        f"{folder_name}IBM_val.csv",
        f"{folder_name}IBM_test.csv",
    )
    multi_window = DataWindow(
        input_width=21,
        label_width=21,
        shift=21,
        train_df=stocks.train_df,
        val_df=stocks.val_df,
        test_df=stocks.test_df,
        label_columns=["Close"],
    )
    val_performance = {}
    performance = {}

    baseline_last = MultiStepLastBaseline(label_index=stocks.column_indices["Close"])
    baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

    val_performance["Baseline - Last"] = baseline_last.evaluate(multi_window.val)
    performance["Baseline - Last"] = baseline_last.evaluate(
        multi_window.test, verbose=0
    )
    multi_window.plot(model=baseline_last, model_name="Baseline - Last")

    baseline_repeat = RepeatBaseline(label_index=stocks.column_indices["Close"])
    baseline_repeat.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    val_performance["Baseline - Repeat"] = baseline_repeat.evaluate(multi_window.val)
    performance["Baseline - Repeat"] = baseline_repeat.evaluate(
        multi_window.test, verbose=0
    )
    multi_window.plot(model=baseline_repeat, model_name="Baseline - Repeat")

    label_index = stocks.column_indices["Close"]
    num_features = stocks.train_df.shape[1]
    with tf.device("/cpu:0"):
        linear = Sequential([Dense(units=1, kernel_initializer=tf.initializers.zeros)])
        history = stocks.compile_and_fit(linear, multi_window)
        # try:
        #     linear.save("linear_model.h5")
        # except Exception as e:
        #     print(e)
        val_performance["Linear"] = linear.evaluate(multi_window.val)
        performance["Linear"] = linear.evaluate(multi_window.test, verbose=0)
        multi_window.plot(model=linear, model_name="Linear")

        dense = Sequential(
            [
                Dense(units=64, activation="relu"),
                Dense(units=64, activation="relu"),
                Dense(units=1, kernel_initializer=tf.initializers.zeros),
            ]
        )
        history = stocks.compile_and_fit(dense, multi_window)
        # try:
        #     dense.save("dense_model.h5")
        # except Exception as e:
        #     print(e)
        val_performance["Dense"] = dense.evaluate(multi_window.val)
        performance["Dense"] = dense.evaluate(multi_window.test, verbose=0)
        multi_window.plot(model=dense, model_name="Dense")

        lstm_model = Sequential(
            [
                LSTM(32, return_sequences=True),
                Dense(1, kernel_initializer=tf.initializers.zeros),
            ]
        )
        history = stocks.compile_and_fit(lstm_model, multi_window)
        # try:
        #     lstm_model.save("lstm_model.h5")
        # except Exception as e:
        #     print(e)
        val_performance["LSTM"] = lstm_model.evaluate(multi_window.val)
        performance["LSTM"] = lstm_model.evaluate(multi_window.test, verbose=0)
        multi_window.plot(model=lstm_model, model_name="LSTM")

        KERNEL_WIDTH = 3
        LABEL_WIDTH = 21
        INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1

        cnn_multi_window = DataWindow(
            input_width=INPUT_WIDTH,
            label_width=LABEL_WIDTH,
            shift=21,
            train_df=stocks.train_df,
            val_df=stocks.val_df,
            test_df=stocks.test_df,
            label_columns=["Close"],
        )

        cnn_model = Sequential(
            [
                Conv1D(
                    32,
                    kernel_size=(KERNEL_WIDTH),
                    activation="relu",
                ),
                Dense(units=32, activation="relu"),
                Dense(units=1, kernel_initializer=tf.initializers.zeros),
            ]
        )
        history = stocks.compile_and_fit(cnn_model, cnn_multi_window)
        try:
            cnn_model.save("../stocks/cnn_model.h5")
        except Exception as e:
            print(e)
        val_performance["CNN"] = cnn_model.evaluate(cnn_multi_window.val)
        performance["CNN"] = cnn_model.evaluate(cnn_multi_window.test, verbose=0)
        cnn_multi_window.plot(model=cnn_model, model_name="CNN")

        cnn_lstm_model = Sequential(
            [
                Conv1D(
                    32,
                    kernel_size=(KERNEL_WIDTH),
                    activation="relu",
                ),
                LSTM(32, return_sequences=True),
                Dense(units=1, kernel_initializer=tf.initializers.zeros),
            ]
        )
        history = stocks.compile_and_fit(cnn_lstm_model, cnn_multi_window)
        # try:
        #     cnn_lstm_model.save("cnn_lstm_model.h5")
        # except Exception as e:
        #     print(e)
        val_performance["CNN + LSTM"] = cnn_lstm_model.evaluate(cnn_multi_window.val)
        performance["CNN + LSTM"] = cnn_lstm_model.evaluate(
            cnn_multi_window.test, verbose=0
        )
        cnn_multi_window.plot(model=cnn_lstm_model, model_name="CNN + LSTM")

        AR_LSTM = AutoRegressive(
            units=32,
            out_steps=21,
            train_df=stocks.train_df,
            val_df=stocks.val_df,
            test_df=stocks.test_df,
        )
        history = stocks.compile_and_fit(AR_LSTM, multi_window)
        try:
            AR_LSTM.save("AR_LSTM", save_format="tf")
        except Exception as e:
            print(e)
        val_performance["ARLSTM"] = AR_LSTM.evaluate(multi_window.val)
        performance["ARLSTM"] = AR_LSTM.evaluate(multi_window.test, verbose=0)
        multi_window.plot(model=AR_LSTM, model_name="ARLSTM")

        mae_val = [v[1] for v in val_performance.values()]
        mae_test = [v[1] for v in performance.values()]

        x = np.arange(len(performance))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(
            x - 0.15,
            mae_val,
            width=0.25,
            color="black",
            edgecolor="black",
            label="Validation",
        )
        ax.bar(
            x + 0.15,
            mae_test,
            width=0.25,
            color="white",
            edgecolor="black",
            hatch="/",
            label="Test",
        )
        ax.set_ylabel("Mean absolute error")
        ax.set_xlabel("Models")

        font_prop = font_manager.FontProperties(size=20)

        for index, value in enumerate(mae_val):
            plt.text(
                x=index - 0.15,
                y=value + 0.005,
                s=str(round(value, 3)),
                ha="center",
                fontproperties=font_prop,
            )

        for index, value in enumerate(mae_test):
            plt.text(
                x=index + 0.15,
                y=value + 0.0025,
                s=str(round(value, 3)),
                ha="center",
                fontproperties=font_prop,
            )

        # plt.ylim(0, 0.33)
        plt.xticks(ticks=x, labels=performance.keys())
        plt.legend(loc="best")

        plt.legend(fontsize=25)  # using a size in points

        plt.tight_layout()
        plt.figure(figsize=(10, 6))
        plt.show()
    try:
        model_names = [
            "Baseline-Last",
            "Baseline-Repeat",
            "Linear",
            "Dense",
            "LSTM",
            "CNN",
            "CNN+LSTM",
            "ARLSTM",
        ]
        data = {"Test - MAE": mae_test, "Validation - MAE": mae_val}
        df = pd.DataFrame(data, index=model_names)
        df_sorted = df.sort_values(by="Test - MAE", ascending=True)
        print(df_sorted.T)
        # df.to_csv("results.csv", index=True, header=True)
    except Exception as e:
        print(e)
