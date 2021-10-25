# Basic
import numpy as np
import scipy.stats as stats

# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

# Supervised
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from category_encoders import TargetEncoder, LeaveOneOutEncoder, BinaryEncoder, OrdinalEncoder
import lightgbm as lgb

# Custom
from consts import Constants


class Algorithms:
    """
    Machine leraning algorithms
    """

    @staticmethod
    def get_alogirthm(algorithm):
        """
        Returns algorithm for given name
        """
        if algorithm == Constants.LOGISTIC_REGRESSION:
            return Algorithms.logistic_regression()
        elif algorithm == Constants.RANDOM_FORSET:
            return Algorithms.random_forest()
        elif algorithm == Constants.LIGHT_GRADIENT_BOOSTING_MACHINE:
            return Algorithms.light_gradient_boosting_machine()
        else:
            raise ValueError('Not implemented')

    @staticmethod
    def logistic_regression():
        """
        Logistic Regression model
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
                    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
        """
        # Define workflow
        pipeline = Pipeline([
            ('encode_categ', TargetEncoder(cols=Constants.CATEG_COLUMNS)),
            ('encode_id', LeaveOneOutEncoder(cols=Constants.ID_COLUMNS)),
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest()),
            #('pca', PCA()),
            ('estimator', LogisticRegression(multi_class='auto', n_jobs=-1, verbose=1))
        ])
        
        # Define Hyperparamter space
        hparams = [{
            'kbest__score_func': [f_classif, mutual_info_classif],         # scoring function for feature selection
            'kbest__k': np.arange(20, 30),                                   # number of feature to select with best score

            #'pca__n_components': np.arange(3, 8),                       # number of principal axes in feature space to keep

            'estimator__solver':['newton-cg', 'saga', 'sag', 'lbfgs'],       # Algorithm to use in the optimization problem
            'estimator__max_iter': np.arange(505, 720),                         # Maximum number of iterations taken for the solvers to converge
            'estimator__penalty': ['l2', 'none'],                            # Used to specify the norm used in the penalization
            'estimator__C': stats.loguniform(1e-5, 150),                           # Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
            'estimator__tol': stats.loguniform(1e-8, 1e-6),                  # Tolerance for stopping criteria
            'estimator__class_weight': ['balanced']                     # Weights associated with classes
        }]

        return pipeline, hparams

    @staticmethod
    def random_forest():
        """
        Random Forest
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        # Define workflow
        pipeline = Pipeline([
            ('encode_categ', TargetEncoder(cols=Constants.CATEG_COLUMNS)),
            ('encode_id', TargetEncoder(cols=Constants.ID_COLUMNS)),
            #('scaler', StandardScaler()),       # required for PCA
            #('pca', PCA()),
            ('kbest', SelectKBest()),
            ('estimator', RandomForestClassifier(verbose=1, n_jobs=-1, warm_start=False))
        ])
        
        # Define Hyperparamter space
        hparams = [{
            #'pca__n_components': np.arange(15, 25),                             # desired dimensionality of output data

            'kbest__score_func': [f_classif, mutual_info_classif],            # scoring function for feature selection
            'kbest__k': np.arange(27, 30),                                      # number of feature to select with best score

            'estimator__n_estimators': np.arange(420, 470),                     # the number of trees in the forest
            'estimator__criterion': ['entropy', 'gini'],                        # the function to measure the quality of a split
            'estimator__max_depth': np.arange(5, 30),                         # the maximum depth of the tree
            #'estimator__min_samples_split': np.arange(3, 70),                 # the minimum number of samples required to split an internal node
            #'estimator__min_samples_leaf': np.arange(2, 20),                  # the minimum number of samples required to be at a leaf node
            'estimator__max_features': ['sqrt', 'log2'],                        # the number of features to consider when looking for the best split
            'estimator__bootstrap': [False],                              # whether bootstrap samples are used when building trees, if False, the whole dataset is used to build each tree
            'estimator__max_samples': stats.uniform(0.5, 0.45),                 # if bootstrap is True, the number of samples to draw from X to train each base estimator
            'estimator__class_weight': ['balanced', 'balanced_subsample', None]
        }]

        return pipeline, hparams

    @staticmethod
    def light_gradient_boosting_machine():
        """
        Light Gradient Boosting Machine
            Source: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf
                    https://lightgbm.readthedocs.io/en/latest/Parameters.html
            GPU Installation: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
                              https://stackoverflow.com/questions/60360750/lightgbm-classifier-with-gpu
                              https://pypi.org/project/lightgbm/
        """
        # Define workflow
        pipeline = Pipeline([
            ('encode_categ', LeaveOneOutEncoder(cols=Constants.CATEG_COLUMNS)),
            ('encode_id', LeaveOneOutEncoder(cols=Constants.ID_COLUMNS)),
            #('scaler', StandardScaler()),
            #('pca', PCA()),
            ('kbest', SelectKBest()),
            ('estimator', lgb.LGBMClassifier(
                n_jobs=-1, 
                device='gpu',
                objective='binary',
                num_class=1))
        ])
        
        # Define Hyperparamter space
        hparams = [{  
            #'pca__n_components': np.arange(90, 100),                     # desired dimensionality of output data
            
            'kbest__score_func': [f_classif, mutual_info_classif],      # scoring function for feature selection
            'kbest__k': np.arange(25, 30),                                                                                                                                                                                                              # number of feature to select with best score

            'estimator__metric': ['binary_error', 'binary_logloss'],       # metric(s) to be evaluated on the evaluation set(s)
            'estimator__boosting_type': ['gbdt', 'goss', 'dart'],
            'estimator__n_estimators': np.arange(290, 350),               # the number of trees in the forest
            'estimator__learning_rate': stats.loguniform(0.005, 0.1),     # shrinkage rate
            #'estimator__linear_tree': [True, False],                     # fit piecewise linear gradient boosting tree
            'estimator__extra_trees': [True, False],                 # use extremely randomized trees, can be used to deal with over-fitting
            'estimator__xgboost_dart_mode': [True, False],              # set this to true, if you want to use xgboost dart mode

            # Size of tree
            #'estimator__num_leaves' : np.arange(135, 170),                  # max number of leaves in one tree
            #'estimator__min_data_in_leaf': np.arange(160, 180),           # minimal number of data in one leaf, can be used to deal with over-fitting
            'estimator__max_depth': np.arange(5, 25),                   # the maximum depth of the tree
            
            # Features
            'estimator__max_bin': np.arange(225, 325),                    # max number of bins that feature values will be bucketed in
            'estimator__min_data_in_bin': np.arange(20, 120),              # minimal number of data inside one bin. Use this to avoid one-data-one-bin (potential over-fitting)
            'estimator__feature_fraction': stats.uniform(0.65, 0.35),      # LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0                                                                                            
            'estimator__bagging_freq': np.arange(2, 5),                  # 0 means disable bagging; k means perform bagging at every k iteration                                               
            'estimator__bagging_fraction': stats.uniform(0.65, 0.35),      # this will randomly select part of data without resampling                                                               
            
            # Regularization
            'estimator__early_stopping_round': np.arange(10, 41),        # will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds, <= 0 means disable
            'estimator__lambda_l1': stats.uniform(0, 0.1),                # L1 regularization
            'estimator__lambda_l2': stats.uniform(0, 0.1),                # L2 regularization
            'estimator__drop_rate': stats.uniform(0, 0.1),            # used only in dart; dropout rate: a fraction of previous trees to drop during the dropout
            'estimator__skip_drop': stats.uniform(0, 0.8),              # used only in dart; probability of skipping the dropout procedure during a boosting iteration
            'estimator__max_drop': np.arange(40, 60),                   # max number of dropped trees during one boosting iteration
        }]

        return pipeline, hparams
