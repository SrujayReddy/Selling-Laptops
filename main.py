import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class UserPredictor():
    def __init__(self):
        # We'll define our columns at initialization
        # and set up our pipeline inside fit.
        self.model_pipeline = None
        self.feature_columns = ['age', 'past_purchase_amt', 'total_time_spent', 'page_visit_count']
        
    def add_logs_as_features(self, users_df, logs_df):
        # Aggregate logs differently: total time and page visits per user
        logs_aggregated = logs_df.groupby('user_id').agg(
            total_time_spent=('seconds', 'sum'),
            page_visit_count=('url', 'count')
        ).reset_index()
        
        # Merge aggregated logs into the users dataframe
        merged_df = users_df.merge(logs_aggregated, on='user_id', how='left')

        # Fill missing values for users with no logs
        merged_df['total_time_spent'] = merged_df['total_time_spent'].fillna(0)
        merged_df['page_visit_count'] = merged_df['page_visit_count'].fillna(0)

        return merged_df

    def fit(self, train_users, train_logs, train_y):
        # Prepare training data
        train_data = self.add_logs_as_features(train_users, train_logs)
        X = train_data[self.feature_columns]
        y = train_y['y']

        # Build a pipeline with scaling and logistic regression
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(random_state=320))
        ])

        # Fit the model
        self.model_pipeline.fit(X, y)

        # Optional cross-validation printout (for debugging, can be commented out)
        scores = cross_val_score(self.model_pipeline, X, y, cv=5)
        # print(f"Cross-Val AVG: {scores.mean()}, STD: {scores.std()}")

    def predict(self, test_users, test_logs):
        # Apply same feature generation
        test_data = self.add_logs_as_features(test_users, test_logs)
        X_test = test_data[self.feature_columns]
        
        # Return boolean predictions (True if predicted 1, else False)
        predictions = self.model_pipeline.predict(X_test)
        return predictions.astype(bool)

