
import time as t
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

from consts import Constants
from utilities import UtilsManager


class PlotManager:
    """
    Plot Manager
    """

    def __init__(self):
        super().__init__()
        return


    def create_palette_warmcold(self, y):
        """
        Create warm-cold plattete
        """
        # Split on unique values
        unique_y = np.unique(y)

        # Calculate colors
        color_palette = np.array(sns.color_palette('coolwarm', n_colors=len(unique_y)))

        palette_dict = {}
        for i, y_val in enumerate(unique_y):
            palette_dict[y_val] = color_palette[i]

        palette = []
        for y_val in y:
            palette += [palette_dict[y_val]]

        return palette

    def plot_distrib_classes(self, dataset):
        """
        Plot distribution of classes
        """
        # Distribution
        class_label = Constants.CLASS_LABEL
        distrib = dataset[class_label].value_counts().sort_index()
        print('Distribution:', distrib, sep='\n')

        # Define values
        x = np.arange(len(distrib))
        y = distrib.values
        palette = self.create_palette_warmcold(y)

        # Plot figure
        axes = sns.barplot(x=x, y=y, palette=palette, edgecolor='gray')
        axes.set_title('Distribution', fontsize=14, pad=10)
        axes.set_ylabel('Instances')
        axes.set_xticklabels(distrib.index, rotation=0)
        axes.set_xlabel(class_label)
        
        # Show
        plt.show()
        plt.close()
        return

    def plot_feature_distrib(self, dataset, bins=150):
        """
        Feature distribution
        """
        for col in dataset.columns:
            plt.figure()
            sns.histplot(data=dataset, x=col, bins=bins, kde=True)
            #plt.figure()
            #sns.boxplot(data=dataset, x=col)
            plt.show()
            plt.close()
        return

    def corr_feature_target(self, dataset):
        """
        Return Pearson correlation between features and target value
        Interpretation: The higer correlation means the class id is higer and vice versa 
        Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        """
        class_label = Constants.CLASS_LABEL
        y_data = dataset[class_label]
        X_data = dataset.drop(class_label, axis=1)

        features_corrs = []
        num_features = X_data.shape[1]
        features = np.arange(num_features)

        # Calculate correlation for each feature
        for col in range(num_features):
            feature = X_data.iloc[:, col].to_numpy()
            corr, _ = stats.pearsonr(feature, y_data)
            features_corrs.append(corr)

        # Info
        print('Correlations Feature-Target:\n', features_corrs)
        
        # Define values
        y = features_corrs
        palette = self.create_palette_warmcold(y)

        # Plot
        _, axes = plt.subplots(figsize=(100, 70), dpi=100)
        axes = sns.barplot(x=features, y=features_corrs, palette=palette, edgecolor=None, ax=axes)
        axes.set_xlabel('Feature')
        axes.set_ylabel('Pearson coef to target')
        axes.set_xticklabels(X_data.columns, rotation=90, fontsize=12)
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        plt.close()
        return

    def corr_feature_matrix(self, dataset):
        """
        Correlation matrix between features from given dataset
        """
        # Plot
        _, axes = plt.subplots(figsize=(120, 80), dpi=100)
        sns.heatmap(
            data=dataset.corr().round(2),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True,
            annot=True,
            linewidths=0.1, 
            vmax=1.0, 
            linecolor='white',
            cbar_kws={'shrink': 0.8},
            annot_kws={'fontsize': 6},
            ax=axes
        )

        # Settings
        axes.set_title('Pearson Correlation Matrix of features', fontsize=14, pad=20)
        axes.tick_params(axis='both', which='both', length=0)
        axes.set_yticklabels(axes.get_yticklabels(), rotation=-360, fontsize=9)
        axes.set_xticklabels(axes.get_xticklabels(), rotation=90, fontsize=9)

        # Show
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.subplots_adjust(bottom=0.2)
        plt.show()
        plt.close()
        return


    def plot_data_pca(self, X_data, y_data=None, num_dim=3, norm=True, title='Visualize dataset', txtlabel='Classes', show=True, num_classes=None, use_tsne=False):
        """
        Visualize hidh-dimensional data into 2D and 3D
            - Visualize with PCA dimension reduction
                Source: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
                        https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
                        Importance of scaling before PCA: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
        """
        # Check dimensions and format + Standardization
        if num_dim < 1:
            raise ValueError('Invalid number of dimension')

        if y_data is not None:
            if not (type(y_data) is np.ndarray):
                y_data = y_data.values.ravel()
            if norm:
                # Standardization
                X_data, y_data = UtilsManager.standardize_data(X_data, y_data)
                
                # Feature selection
                if Constants.VIZ_SELECT_K > 1:
                    selector = SelectKBest(score_func=f_classif, k=Constants.VIZ_SELECT_K)
                    X_data = selector.fit_transform(X=X_data, y=y_data)
        else:
            X_data = UtilsManager.standardize_data(X_data)

        # PCA
        pca = PCA(n_components=num_dim)
        X_data_reduced = pca.fit_transform(X_data)
        d0 = X_data_reduced[:, 0]
        d1 = np.zeros_like(d0) if num_dim <= 1 else X_data_reduced[:, 1]
        d2 = np.zeros_like(d0) if num_dim <= 2 else X_data_reduced[:, 2]

        if show:
            self.plot_explained_variance(pca, num_dim=num_dim, show=show)
        
        if num_dim > 2:
            self.plot_3d_data(d0, d1, d2, y_data=y_data, num_classes=num_classes, title=title, txtlabel=txtlabel, show=show)
        else:
            self.plot_2d_data(d0, d1, y_data=y_data, num_classes=num_classes, title=title, txtlabel=txtlabel, show=show)

        # t-SNE
        if use_tsne:
            self.plot_tsne(X_data, y_data, num_dim=num_dim, num_classes=num_classes, txtlabel=txtlabel, title=title, show=show)
        return

    def plot_tsne(self, X_data, y_data, num_dim, num_classes, txtlabel, title, show):
        """
        Plot feature space with t-SNE method
        """
        title = 'tSNE_' + title
        if X_data.shape[1] > Constants.TSNE_LIMIT:            # Reduce number of feature with PCA to reduce t-SNE computation
            pca = PCA(n_components=Constants.TSNE_LIMIT)
            X_data = pca.fit_transform(X_data)

        tsne = TSNE(n_components=num_dim, n_jobs=-1, n_iter=Constants.TSNE_NUM_ITER)
        X_data_reduced = tsne.fit_transform(X_data)

        d0 = X_data_reduced[:, 0]
        d1 = np.zeros_like(d0) if num_dim <= 1 else X_data_reduced[:, 1]
        d2 = np.zeros_like(d0) if num_dim <= 2 else X_data_reduced[:, 2]

        if num_dim > 2:
            self.plot_3d_data(d0, d1, d2, y_data=y_data, num_classes=num_classes, title=title, txtlabel=txtlabel, show=show)
        else:
            self.plot_2d_data(d0, d1, y_data=y_data, num_classes=num_classes, title=title, txtlabel=txtlabel, show=show)
        return

    def plot_explained_variance(self, pca, num_dim=3, show=True):
        """
        Plot percentages of variance explained by each of the selected PCA components
        """
        print('Percentage of variance explained by each of the selected components:\n', pca.explained_variance_ratio_)
        x = np.arange(1, num_dim+1)
        y = np.cumsum(pca.explained_variance_ratio_)
        fig, axes = plt.subplots(1, 1)
        axes = sns.lineplot(x=x, y=y, marker='o', ax=axes) 
        axes.set_xticks(x)
        axes.set_xticklabels(x)
        axes.set_title('Explained Variance for each dimension', fontsize=14)
        axes.set_xlabel('Number of dimensions', fontsize=10)
        axes.set_ylabel('Cumulative explained variance', fontsize=10)
        if show:
            plt.show()
            plt.close()
        return

    def plot_2d_data(self, d0, d1, y_data=None, num_classes=None, title='2D Chart', txtlabel='Classes', show=True):
        """
        Plot data in 2D space
            Warning: Only works with consecutive numbers assigned to classes 
        """
        if y_data is not None:
            if num_classes is None:
                num_classes = len(np.unique(y_data))
            colors = Constants.COLORS
            cmap = mpl.colors.ListedColormap(colors)                  # Define the colormap
            min_val = y_data.min()
            max_val = y_data.max() + 1
            bounds = np.linspace(min_val, max_val, num_classes+1)     # Define the bins and normalize
            norm = BoundaryNorm(bounds, cmap.N)

        fig, axes = plt.subplots(1, 1)
        axes.set_title(title)
        axes.set_xlabel('D0', fontsize=10)
        axes.set_ylabel('D1', fontsize=10)
        if y_data is not None and num_classes > 1:
            scatter_plot = axes.scatter(x=d0, y=d1, c=y_data, cmap=cmap, s=10, norm=norm, alpha=0.6)
            cbar = plt.colorbar(scatter_plot, spacing='proportional', ticks=bounds)                     # Create the colorbar
            labels = np.arange(min_val, max_val, 1)
            cbar.set_ticklabels(labels)
            cbar.set_ticks(labels + 0.5)
            cbar.set_label(txtlabel)
        else:
            scatter_plot = axes.scatter(x=d0, y=d1, s=10, alpha=0.6)

        # Save and show
        if show:
            plt.show()
            plt.close()
        return

    def plot_3d_data(self, d0, d1, d2, y_data=None, num_classes=None, title='3D Chart', txtlabel='Classes', show=True):
        """
        Plot data in 3D space
        """
        if y_data is not None:
            if num_classes is None:
                num_classes = len(np.unique(y_data))
            colors = Constants.COLORS
            cmap = mpl.colors.ListedColormap(colors)                 # Define the colormap
            min_val = y_data.min()
            max_val = y_data.max() + 1
            bounds = np.linspace(min_val, max_val, num_classes+1)     # Define the bins and normalize
            norm = BoundaryNorm(bounds, cmap.N)

        fig, axes = plt.subplots(1, 1)
        #axes = fig.add_axes([0.85, 0.16, 0.02, 0.65])  # left, bottom, right, top
        axes.get_xticklabels([])
        axes.get_yticklabels([])
        plt.axis('off')
        ax = Axes3D(fig, elev=-150, azim=110, title=title)
        if y_data is not None and num_classes > 1:
            scatter_plot = ax.scatter(d0, d1, d2, c=y_data, cmap=cmap, edgecolor='k', norm=norm, s=40, alpha=0.6)
            #cbar = mpl.colorbar.ColorbarBase(axes, cmap=cmap, norm=norm, ticks=bounds, boundaries=bounds, orientation='vertical', drawedges=False)
            cbar = plt.colorbar(scatter_plot, spacing='proportional', ticks=bounds, ax=axes)
            labels = np.arange(min_val, max_val, 1)
            cbar.set_ticklabels(labels)
            cbar.set_ticks(labels + 0.5)
            cbar.set_label(txtlabel)
        else:
            scatter_plot = ax.scatter(d0, d1, d2, edgecolor='k', s=40, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('D1')
        ax.set_ylabel('D2')
        ax.set_zlabel('D3')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        # Save and show
        if show:
            plt.show()
            plt.close()
        return


    def visualize_data(self, dataset):
        """
        Visualization of given dataset
        """
        # Transfrom categorical features to numerical
        dataset_num = UtilsManager.conv_to_numerical(dataset, Constants.CATEG_COLUMNS)

        # Classes distribution
        self.plot_distrib_classes(dataset_num)

        # Feature distribution
        self.plot_feature_distrib(dataset_num)

        # Correlation feature-target
        self.corr_feature_target(dataset_num)

        # Correlation between features
        self.corr_feature_matrix(dataset_num)

        # Feature space
        X_data = dataset_num.iloc[:, :-1]
        y_data = dataset_num.iloc[:, -1:]
        self.plot_data_pca(X_data, y_data, use_tsne=False, norm=True)
        return


    def confusion_matrix(self, conf_matrix, title=''):
        """
        Plot given confusion matrix in dataframe format as heatmap
        """
        # Plot confusion matrix as heatmap
        plt.figure()
        axes = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        axes.tick_params(axis='both', which='both', length=0)
        axes.set_xticklabels(axes.get_xticklabels(), rotation=-360, horizontalalignment='center')
        axes.set_title(title + ' Confusion matrix', fontsize=18, pad=20)
        axes.set_ylabel('Actual', fontsize=14)
        axes.set_xlabel('Predicted', fontsize=14)
        axes.xaxis.labelpad = 20

        # Show plot
        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()
        plt.show()
        plt.close()
        return

    def plot_roc_curve(self, y_test, y_preds, model_name, pos_label=1):
        """
        Plot ROC curve
            It will also change the number of FPR and TRP by changing the treshold (all similar like precision-recall curve)
            One treshold value correspond to one FPR and TRP value, except some treshold can be removed because they are not relevant for plotting the curve
            Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        """
        # Calculate curve
        fpr, tpr, thresholds = roc_curve(y_test, y_preds, pos_label=pos_label)
        auc_score = roc_auc_score(y_test, y_preds)

        # Default Threshold
        threshold_value = abs(pos_label)                                        # by setting temp value you can manupilate with treshold and call make_predictions function
        def_threshold_ind = np.argmin(np.abs(thresholds - threshold_value))     # return index of aprox. default treshold value (for probability and score)
        fpr_def_threshold = fpr[def_threshold_ind]
        tpr_def_threshold = tpr[def_threshold_ind]

        # Plot curve
        plt.figure()
        plt.title('ROC curve [+class={}]'.format(pos_label), fontsize=16)
        plt.xlabel('False Positve Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.plot(fpr, tpr, lw=3, label='{}; AUC={:.2f}'.format(model_name, auc_score))
        plt.plot([0,1], [0,1], lw=3, color='navy', linestyle='--', label='Random classifier; AUC=0.5', alpha=0.7)
        plt.plot(fpr_def_threshold, tpr_def_threshold, marker='o', markersize=10, fillstyle='none', c='r', mew=3, alpha=0.8)
        #plt.axes().set_aspect('equal')
        plt.legend(loc='best')
        plt.show()
        plt.close()
        return

    def feature_importance(self, features, importances, sort=True, title=None):
        """
        Feature importance as bar plot
        """
        # Info
        print(importances)
        x_features = np.arange(len(features))
        
        # Sort Numeric Features
        if sort:
            sorted_order = features.argsort()
            features = features[sorted_order]
            importances = importances[sorted_order]

        # Define values
        y = importances
        palette = self.create_palette_warmcold(y)

        # Plot
        axes = sns.barplot(x=x_features, y=importances, palette=palette, edgecolor=None)
        axes.set_xticklabels(features, rotation=90)
        axes.set_title(title, fontsize=18, pad=20)
        axes.set_xlabel('Feature', fontsize=12)
        axes.set_ylabel('Importance', fontsize=12)
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        plt.close()
        return


    def visualize_model(self, y_test, y_preds, pipeline, algorithm, features):
        """
        Visualization of model performance
        """
        # ROC curve
        self.plot_roc_curve(y_test, y_preds, algorithm)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_preds)
        self.confusion_matrix(conf_matrix, title=algorithm)

        # Get selected features
        selected_features, estimator = UtilsManager.get_selected_features(features, pipeline)
        if selected_features is not features:
            print('Selected features:\n', selected_features)

        # Feature importance
        if algorithm in (Constants.RANDOM_FORSET):
            importances = estimator.feature_importances_
            self.feature_importance(selected_features, importances, title=algorithm)
            
        elif algorithm in (Constants.LOGISTIC_REGRESSION):
            classes = sorted(np.unique(y_test))
            if len(classes) > 2:
                for i, id_class in enumerate(classes):      # Multi-class
                    importances = estimator.coef_[i]
                    title = f'Class {id_class} - { algorithm}'
                    self.feature_importance(selected_features, importances, title=title)
            else:                                           # Binary
                importances = estimator.coef_[0]
                title = f'Coef {algorithm}'
                self.feature_importance(selected_features, importances, title=title)
        return


if __name__ == "__main__":
    mngr = PlotManager()