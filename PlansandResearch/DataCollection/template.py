import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from Visualizations import Visualizer
from HyperParameterTuner import HyperParameterTuner

# imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump, load

"""
    Use if you want
    rename the class to reflect the data you are using
    have relevant file and folder names
    Dont forrget to remove the NotImplementedError
"""


class TrainModelTemplate:
    def __init__(self, folder_name: str, file_name: str, target: str):
        self.folder_name = folder_name
        self.file_name = file_name
        self.data = self.load_data()
        self.train_data, self.test_data = self.split_data()
        self.train_data, self.validation_data = self.split_validataion_data(
            validation_split=0.2
        )
        self.train_labels, self.test_labels, self.validation_labels = self.split_labels(
            target=target
        )

    def load_data(self) -> pd.DataFrame:
        """Load data from the data folder"""
        return pd.DataFrame(pd.read_csv(self.file_name))
        # raise NotImplementedError

    def split_data(self) -> pd.DataFrame:
        """Split data into train and test
        Returns: two dataframes: train and test data via return train_df, test_df
        Probably 90% train and 10% test
        """
        return train_test_split(self.data, test_size=0.1, random_state=42)
        # raise NotImplementedError

    def split_validataion_data(self, validation_split: float) -> pd.DataFrame:
        """Split train data into train and validation
        Returns: two dataframes: train and validation data
        Usually 80% train and 20% validation
        """
        return train_test_split(
            self.train_data, test_size=validation_split, random_state=42
        )
        # raise NotImplementedError

    def split_labels(self, target: str) -> pd.DataFrame:
        """Split labels from the data
        Returns: two dataframes: train and test labels
        """
        train_labels = self.train_data.pop(target)
        test_labels = self.test_data.pop(target)
        validation_labels = self.validation_data.pop(target)
        return train_labels, test_labels, validation_labels
        # raise NotImplementedError

    def linear_regression(self) -> None:
        model = LinearRegression()
        model.fit(self.train_data, self.train_labels)
        y_pred = model.predict(self.test_data)
        # Visualizer.plot_loss_plt(self.folder_name, self.file_name, model)
        print("Linear Regression R2 Score: ", r2_score(self.test_labels, y_pred))
        # raise NotImplementedError

    def decision_tree(self) -> None:
        """Decision tree model"""
        model = DecisionTreeRegressor()
        model.fit(self.train_data, self.train_labels)
        y_pred = model.predict(self.test_data)
        print("Decision Tree R2 Score: ", r2_score(self.test_labels, y_pred))

    def random_forest(
        self,
        n_estimators: int = 500,
        max_depth: int = 40,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        """Random forest model"""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        model.fit(self.train_data, self.train_labels)
        y_pred = model.predict(self.test_data)
        print("Random Forest R2 Score: ", r2_score(self.test_labels, y_pred))

    def neural_network(self) -> None:
        """Neural network model"""
        model = tf.keras.Sequential(
            [
                layers.Dense(
                    64, activation="relu", input_shape=(self.train_data.shape[1],)
                ),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),  # Assuming a regression task
            ]
        )

        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(
            self.train_data,
            self.train_labels,
            epochs=10,
            batch_size=32,
            validation_data=(self.validation_data, self.validation_labels),
        )

        y_pred = model.predict(self.test_data)
        print("Neural Network R2 Score: ", r2_score(self.test_labels, y_pred))

    def dnn(self) -> None:
        """Deep neural network model"""
        hyperparameter_tuner = HyperParameterTuner(self)
        best_hps = hyperparameter_tuner.kt_dnn_tuner()

        model = hyperparameter_tuner.dnn_model_builder(best_hps)

        history = model.fit(
            self.train_data,
            self.train_labels,
            epochs=100,  # You can adjust the number of epochs
            validation_data=(self.validation_data, self.validation_labels),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25)
            ],
        )

        y_pred = model.predict(self.test_data)
        print("DNN R2 Score: ", r2_score(self.test_labels, y_pred))

        # Visualize the training loss
        visualizer = Visualizer()
        visualizer.plot_loss_plt(self.folder_name, self.file_name, history)
        # raise NotImplementedError

        model.save("EmployeeAttrition.h5")


if __name__ == "__main__":
    # 1. Load data
    # 2. Split data
    # 3. Split labels
    # 4. Train models
    # 5. Visualize results
    # 6. Save results

    # folder_name = "data/"
    # file_name = "PlansandResearch/DataCollection/bankruptcy_predictor_cleaned.csv"
    # model = TrainModelTemplate(folder_name, file_name, "Bankrupt?")
    # model.linear_regression()
    # model.decision_tree()
    # model.random_forest()
    # dump(model, "Bankruptcy_rf.joblib")
    # model.neural_network()
    # model.dnn()

    folder_name = "data/ea/"
    file_name = "PlansandResearch/DataCollection/employee_attrition_cleaned.csv"
    model = TrainModelTemplate(folder_name, file_name, "Attrition")
    model.linear_regression()
    model.decision_tree()
    model.random_forest()
    dump(model, "EmployeeAttrition_rf.joblib")
    model.neural_network()
    model.dnn()
