import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np

def logistic(x):
    return 1 / (1 + np.exp(-x))

class PatientAveragingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, pipeline, id_column='patient_id'):
        self.pipeline = pipeline
        self.id_column = id_column

    def fit(self, X, y):
        # No usar patient_id como input de entrenamiento
        if self.id_column in X.columns:
            X_ = X.drop(columns=[self.id_column])
        else:
            X_ = X
        self.pipeline.fit(X_, y)
        return self

    def predict_proba(self, X):
        has_id = self.id_column in X.columns
        X_ = X.drop(columns=[self.id_column]) if has_id else X
        probs = self.pipeline.predict_proba(X_)

        if not has_id:
            return probs  # Devolver directamente si no hay patient_id

        # Promediar los scores por patient_id
        ids = X[self.id_column].values
        prob_df = pd.DataFrame(probs)
        prob_df[self.id_column] = ids
        from scipy.stats import trim_mean
        from scipy.special import logit
        # Define how much to trim (e.g., 10% from each tail → total 20%)
        
        avg_probs = prob_df.groupby(self.id_column).median().sort_index()
        #trim_fraction = 0.1
        # Apply trimmed mean per group
        # avg_probs = (
        #     prob_df
        #     .groupby(self.id_column)
        #     .agg(lambda x: trim_mean(x, proportiontocut=trim_fraction))
        #     .sort_index()
        # )
        # print("a")
        #print([logit(x) for x in (3*avg_probs.values)])
        return avg_probs.values

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)



import plotly.express as px

# Create a DataFrame for feature importances
feature_importances_df = pd.DataFrame({
    'Feature': X_train_val.columns,
    'Importance': (importances[1:])
})
# Sort the DataFrame by importance
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
# Create a bar plot using Plotly
fig = px.bar(
    feature_importances_df.head(50),
    x='Importance',
    y='Feature',
    orientation='h',
    title='Top 50 Feature Importances',
    labels={'Importance': 'Importance', 'Feature': 'Feature'},
    color='Importance',
    color_continuous_scale='Viridis'
)
fig.show()