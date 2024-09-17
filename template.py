import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from Visualizations import Visualizer
# imports

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
        self.train_labels, self.test_labels = self.split_labels(target=target)

    def load_data(self) -> pd.DataFrame:
        """Load data from the data folder"""
        raise NotImplementedError

    def split_data(self) -> pd.DataFrame:
        """Split data into train and test
        Returns: two dataframes: train and test data via return train_df, test_df
        Probably 90% train and 10% test
        """
        raise NotImplementedError

    def split_validataion_data(self, validation_split: float) -> pd.DataFrame:
        """Split train data into train and validation
        Returns: two dataframes: train and validation data
        Usually 80% train and 20% validation
        """
        raise NotImplementedError

    def split_labels(self, target: str) -> pd.DataFrame:
        """Split labels from the data
        Returns: two dataframes: train and test labels
        train_labels = train_data.pop(target)
        test_labels = test_data.pop(target)
        return train_labels, test_labels
        """
        raise NotImplementedError

    def linear_regression(self) -> None:
        """Linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            Visualizer.plot_loss_plt(folder_name, file_name, model)
            print("Linear Regression R2 Score: ", r2_score(y_test, y_pred))
        """
        raise NotImplementedError

    def decision_tree(self) -> None:
        """Decision tree model"""
        raise NotImplementedError

    def random_forest(
        self,
        n_estimators: int = 500,
        max_depth: int = 10,
        n_jobs: int = -1,
        random_state: int = None,
    ) -> None:
        """Random forest model"""
        raise NotImplementedError

    def neural_network(self) -> None:
        """Neural network model"""
        raise NotImplementedError

    def dnn(self) -> None:
        """Deep neural network model"""
        raise NotImplementedError


if __name__ == "__main__":
    """
    1. Load data
    2. Split data
    3. Split labels
    4. Train models
    5. Visualize results
    6. Save results

    folder_name = "data/"
    file_name = "data.csv"
    model = TrainModelTemplate(folder_name, file_name, target_col)
    model.linear_regression()
    model.decision_tree()
    model.random_forest()
    model.neural_network()
    model.dnn()
    """
    
