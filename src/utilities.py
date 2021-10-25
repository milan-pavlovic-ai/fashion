
import pandas as pd
import numpy as np

from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

from consts import Constants


class UtilsManager():
    """
    Utilities
    """

    @staticmethod
    def standardize_data(data, y_data=None, save=False, path=None):
        """
        Standardize dataset before visualization
        """
        if y_data is None:
            # Split
            X_data = data.iloc[:, :-1].copy()
            y_data = data.iloc[:, -1:].copy()
            
            # Standardize
            X_data = StandardScaler().fit_transform(X_data)
            X_data = pd.DataFrame(X_data, columns=data.columns[:-1])
            data = pd.concat([X_data, y_data], axis=1)

            # Save
            if save:
                if path is not None:
                    data.to_csv(path, index=False)
                    print('Normalized file is saved: ' + path)
                else:
                    print('File not found!')
                    exit(-1)
            return data
        else:
            # Standardize
            X_data = data.copy()
            X_data = StandardScaler().fit_transform(X_data)
            return X_data, y_data

    @staticmethod
    def conv_to_numerical(dataset, categ_columns):
        """
        Transfrom categorical features to numerical
        """
        # Use Target encoding to encode two categorical features
        encoder = TargetEncoder(cols=categ_columns)

        # Transform the datasets
        X_data = dataset.iloc[:, :-1].copy()
        y_data = dataset.iloc[:, -1:].copy()
        numeric_dataset = encoder.fit_transform(X_data, y_data)
        numeric_dataset[Constants.CLASS_LABEL] = y_data

        return numeric_dataset
             
    @staticmethod
    def get_selected_features(features, pipeline):
        """
        Get selected features
            It works only in order Select K feature -> PCA transform
        """
        # No selection
        estimator = None
        selected_features = features

        # Get pipeline from calibartion wrapper
        if hasattr(pipeline, 'base_estimator'):
            pipeline = pipeline.base_estimator

        # Select features
        for step in pipeline.steps:
            name, transformer = step

            if name == 'kbest':
                scores = transformer.scores_
                k = transformer.get_params()['k']
                k_scores = pd.Series(scores).nlargest(k)
                selected_features = features[k_scores.index]

            elif name == 'pca' or name == 'lsa' or name == 'svd':
                n = transformer.get_params()['n_components']
                selected_features = np.arange(0, n)
                
            elif name == 'estimator':
                estimator = transformer

        return selected_features, estimator