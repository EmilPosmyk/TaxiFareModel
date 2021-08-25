import joblib

from TaxiFareModel.pipeline import TaxiFarePipeline
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, holdout

class Trainer:

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline_data = None
        self.X = X
        self.y = y

    @staticmethod
    def set_pipeline():
        """defines the pipeline as a class attribute"""
        return TaxiFarePipeline.create_pipeline()

    def fit(self):
        '''returns a trained pipelined model'''
        self.pipeline_data.fit(self.X_train, self.y_train)
        # print(self.pipeline)

    def run(self):
        """set and train the pipeline"""

        # get data
        df = get_data()

        # Holdout
        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)

        # Pipeline
        self.pipeline_data = Trainer.set_pipeline()
        print(self.pipeline_data)

        # Fit
        self.fit()

        # self.save_pipeline()  # TODO split encoders into 2 classes ?

    def evaluate(self, x_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''

        y_pred = self.pipeline_data.predict(x_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)

        return rmse

    def save_pipeline(self):
        joblib.dump(self.pipeline_data, 'pipeline.joblib')


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
