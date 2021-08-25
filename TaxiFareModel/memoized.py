from memoized_property import memoized_property

import mlflow
from mlflow.tracking import MlflowClient


class Trainer:

    MLFLOW_URI = "https://mlflow.lewagon.co/"

    def __init__(self):
        self.experiment_name = "test_experiment"

    # method name will be used as a param name !
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    # method name will be used as a param name !
    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # in a loop must be created a new run ! so, NO @memoized_property here !
    #@memoized_property
    def mlflow_create_run(self):
        #return self.mlflow_client.create_run(self.mlflow_experiment_id)
        self.mlflow_run = self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def train(self):

        for model in ["linear", "Randomforest"]:
            self.mlflow_create_run()
            self.mlflow_log_metric("rmse", 6.66)
            self.mlflow_log_param("model", model)
            self.mlflow_log_param("student_name", 'Alien')

    def train_1(self):

        EXPERIMENT_NAME = "test_experiment"

        # Indicate mlflow to log to remote server
        mlflow.set_tracking_uri(self.MLFLOW_URI)

        client = MlflowClient()

        try:
            experiment_id = client.create_experiment(EXPERIMENT_NAME)
        except BaseException:
            experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

        for model in ["linear", "Randomforest"]:
            run = client.create_run(experiment_id)
            client.log_metric(run.info.run_id, "rmse", 6.66)
            client.log_param(run.info.run_id, "model", model)
            client.log_param(run.info.run_id, "student_name", 'disintegrated')


trainer = Trainer()
trainer.train()
#trainer.train_1()  # like in ml_flow_test_lw/py

