from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder


class TaxiFarePipeline:

    @staticmethod
    def create_pipeline():

        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe