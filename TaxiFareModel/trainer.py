import joblib

import mlflow
from mlflow.tracking import MlflowClient

from memoized_property import memoized_property

from TaxiFareModel.pipeline import TaxiFarePipeline
from TaxiFareModel.utils import compute_rmse, time_tracker
from TaxiFareModel.data import get_data, holdout


class Trainer:

    MLFLOW_URI = "https://mlflow.lewagon.co/"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        # replace with your country code, city, github_nickname and model name and version
        self.experiment_name = "[BE] [BXL] [Alien] TaxiFareModel + 1"
        self.pipeline_data = None
        self.X = X  # in test 03 it's X_train
        self.y = y  # in test 03 it's y_train

    def fit(self):
        '''returns a trained pipelined model'''

        # self.pipeline_data.fit(self.X_train, self.y_train)
        self.pipeline_data.fit(self.X, self.y)

        # print(self.pipeline)

    @time_tracker  # displays result from jupiter notebook
    def run(self):
        """set and train the pipeline"""

        # get data (its in jupiter notebook)
        # df = get_data()

        # Holdout (its in jupiter notebook)
        # self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)

        # Pipeline
        self.pipeline_data = Trainer.set_pipeline()
        print(self.pipeline_data)

        # Fit
        self.fit()

        # self.save_pipeline()  # TODO split encoders into 2 classes ?

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''

        y_pred = self.pipeline_data.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)

        return rmse

    def save_pipeline(self):
        joblib.dump(self.pipeline_data, 'pipeline.joblib')

    @staticmethod
    def set_pipeline():
        """defines the pipeline as a class attribute"""
        return TaxiFarePipeline.create_pipeline()

    @staticmethod
    def call_like_notebook(self):

        from sklearn.model_selection import train_test_split
        from TaxiFareModel.data import get_data, clean_data

        N = 10_000
        df = get_data(nrows=N)
        df = clean_data(df)
        y = df["fare_amount"]
        X = df.drop("fare_amount", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        trainer = Trainer(X_train, y_train)
        trainer.run()
        trainer.evaluate(X_test, y_test)
        # trainer.save_pipeline()


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')

    # Trainer.call_like_notebook()

