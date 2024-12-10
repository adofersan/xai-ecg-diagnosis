from typing import OrderedDict
import numpy as np
import pandas as pd
from joblib import dump, load

from metrics_multilabel import (
    multilabel_instance_auc,
    multilabel_macro_auc,
    multilabel_weighted_instance_auc,
    multilabel_weighted_macro_auc,
)


class PTBXLModel:
    def __init__(self, estimator):
        """Initialize the classifier with an estimator."""
        self.__estimator = estimator
        self.__feature_names = None
        self.__target_names = None

    @staticmethod
    def load(model_path):
        """Load a model from a file."""
        return load(model_path)

    def test_and_save(self, name_prefix, X_test, y_test, lh_test):
        y_pred = self.predict_proba(X_test)
        # Metrics
        macro_auc = multilabel_macro_auc(y_test, y_pred)
        weighted_macro_auc = multilabel_weighted_macro_auc(y_test, y_pred, lh_test)
        instance_auc = multilabel_instance_auc(y_test, y_pred)
        weighted_instance_auc = multilabel_weighted_instance_auc(
            y_test, y_pred, lh_test
        )
        # Naming
        estimator_name = self.__estimator.named_steps[
            "model"
        ].estimator.__class__.__name__
        fmacro = lambda x: f"{x:.3f}".split(".")[1]
        filename = (
            f"../models/{name_prefix}_{estimator_name}."
            f"{fmacro(macro_auc)}."
            f"{fmacro(weighted_macro_auc)}."
            f"{fmacro(instance_auc)}."
            f"{fmacro(weighted_instance_auc)}.pkl"
        )
        print(filename)
        dump(self, filename)

    def fit(self, X, y):
        """Fit the model to the given training data."""
        self.__estimator.fit(X, y)
        self.__feature_names = X.columns
        self.__target_names = y.columns

    def get_feature_names(self):
        """Get the feature names."""
        return self.__feature_names

    def get_target_names(self):
        """Get the target names."""
        return self.__target_names

    def predict(self, X):
        """Make predictions using the loaded model."""
        return self.__estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities using the loaded model."""
        proba = self.__estimator.predict_proba(X)
        y_pred = pd.DataFrame([arr[:, 1] for arr in proba]).T
        return y_pred

    def feature_importances(self):
        """Get feature importances or coefficients from the model if available."""
        if hasattr(self.__estimator.named_steps["model"], "estimators_"):
            return self._get_feature_importances()
        elif hasattr(self.__estimator.named_steps["model"], "coef_"):
            return self._get_coefficients()
        else:
            raise AttributeError(
                f"The model does not have feature importances or coefficients."
            )

    def _get_feature_importances(self):
        """Helper method to get feature importances from the model."""
        multi_output_rf = self.__estimator.named_steps["model"]
        importances = OrderedDict()

        feature_names = self.__feature_names.copy()
        for label, estimator in zip(self.__target_names, multi_output_rf.estimators_):
            # Get feature importances
            feature_importances = estimator.feature_importances_

            # Create a list of tuples (feature name, importance)
            estimator_importances = [
                (feature_name, float(importance))
                for feature_name, importance in zip(feature_names, feature_importances)
            ]

            # Add the current estimator's importances to the main dictionary
            importances[label] = estimator_importances

        return importances

    def _get_coefficients(self):
        """Helper method to get coefficients from the model."""
        multi_output_lr = self.__estimator.named_steps["model"]
        coefficients = OrderedDict()

        feature_names = self.__feature_names.copy()
        for label, estimator in zip(self.__target_names, multi_output_lr.estimators_):
            coef = estimator.coef_

            if hasattr(estimator, "intercept_"):
                intercept = estimator.intercept_
                coef = np.insert(coef, 0, intercept, axis=1)
                feature_names = np.insert(feature_names, 0, "intercept")

            class_coefficients = OrderedDict()
            for feature_name, value in zip(feature_names, coef.flatten()):
                class_coefficients[feature_name] = float(value)

            coefficients[label] = class_coefficients

        return coefficients

    def model_type(self):
        """Return the type of the model being used."""
        model_types = OrderedDict()

        if hasattr(self.__estimator.named_steps["model"], "estimators_"):
            for label, estimator in zip(
                self.__target_names, self.__estimator.named_steps["model"].estimators_
            ):
                model_types[label] = type(estimator)
        else:
            model_types["model"] = type(self.__estimator.named_steps["model"])

        return model_types

    def build_shap_explainer(self, shap_explainer):
        """Get SHAP explainers for each estimator in the model."""
        explainers = OrderedDict()

        if hasattr(self.__estimator.named_steps["model"], "estimators_"):
            for label, estimator in zip(
                self.__target_names, self.__estimator.named_steps["model"].estimators_
            ):
                explainer = shap_explainer(estimator)
                explainers[label] = explainer
        else:
            explainer = shap_explainer(self.__estimator.named_steps["model"])
            explainers["model"] = explainer

        return explainers

    def get_encoder(self):
        """Get the encoder from the pipeline if available."""
        if "encoder" in self.__estimator.named_steps:
            return self.__estimator.named_steps["encoder"]
        else:
            raise AttributeError("The pipeline does not contain an encoder step.")
