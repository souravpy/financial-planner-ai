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
