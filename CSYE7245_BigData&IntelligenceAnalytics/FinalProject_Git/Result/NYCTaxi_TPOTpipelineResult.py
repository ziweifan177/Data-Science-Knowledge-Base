import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Score on the training set was:-0.14699252836330726
exported_pipeline = XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=2, n_estimators=100, nthread=1, subsample=1.0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
