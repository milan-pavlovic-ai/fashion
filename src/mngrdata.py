
import time as t
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import scipy.stats as stats

from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from sklearn.preprocessing import KBinsDiscretizer
from fast_ml.utilities import display_all
from fast_ml import eda

from consts import Constants
from mngrplot import PlotManager
from utilities import UtilsManager


class DataManager:
    """
    Data manager
    """

    def __init__(self, info=False, visual=False, sample=None):
        super().__init__()
        self.info = info
        self.visual = visual
        self.sample = sample
        self.mngrplot = PlotManager()

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.training = None
        self.wait_days_median = None
        self.age_median = None
        self.age_mean = None
        self.title_mode = None
        self.item_color_counts = None
        self.item_size_counts = None
        return

    @staticmethod
    def describe_data(df, info=True, duplicates=False, visual=False):
        """
        Describe given dataframe
            Source: https://github.com/ResidentMario/missingno
                    https://pypi.org/project/fast-ml/
        """
        if info:
            # Dataset
            print('\n', df, sep='\n')
            print(df.info())

            # NaNs
            print('NaN:\n', df.isna().sum())
            print('NaN rows:\n', df[df.isna().any(axis=1)])
            print(df.describe(include='all'))

            # Visualization
            if visual:
                msno.matrix(df)
                plt.subplots_adjust(top=0.75)
                plt.show()
                plt.close()

            # Duplicates
            if duplicates:
                for col in df.columns:
                    uni_val = df[col].value_counts()
                    uni_num = len(uni_val)
                    print(f'\n{col} : {uni_num}')
                    if df[col].dtype in (np.int32, np.int64, np.float32, np.int64, np.datetime64, np.dtype('M8[ns]')):
                        print(f'\tmin:\t{df[col].min()}\n\tmax:\t{df[col].max()}\n\tmean:\t{df[col].mean()}\n\tmedian:\t{df[col].median()}')
                    print(uni_val)
        
                dupl = df.duplicated().value_counts()
                #df = df.drop_duplicates(keep='first')
                print('\nDuplicates:\n', dupl)

            # EDA
            summary_info = eda.df_info(df)
            display_all(summary_info)
        return

    # Dates

    def add_date_features(self, dataset, feature):
        """
        Extract new date features from given date feature
        """
        dataset[f'{feature}_month'] = dataset[feature].dt.month
        dataset[f'{feature}_day'] = dataset[feature].dt.day
        dataset[f'{feature}_dayweek'] = dataset[feature].dt.dayofweek
        dataset[f'{feature}_quarter'] = dataset[feature].dt.quarter

        return dataset

    def att_order_date(self, dataset):
        """
        Feature engineering of attribute 'order date'
        """
        # Date feature engineering 
        dataset = self.add_date_features(dataset, 'order_date')
        return dataset

    def att_delivery_date(self, dataset):
        """
        Feature engineering of attribute 'delivery date'
        """
        # Fix invalid by setting median of waiting days
        dataset['invalid_delivery_dates'] = dataset['delivery_date'] < dataset['order_date']

        # Calculate median of waiting days on clean training dataset
        if self.training: 
            clean_dataset = dataset[dataset['invalid_delivery_dates'] == False]
            self.wait_days_median = (clean_dataset['delivery_date'] - clean_dataset['order_date']).median()

        # Set calculated median to invalid delivery dates
        mask = dataset['invalid_delivery_dates']
        dataset.loc[mask, 'delivery_date'] = dataset.loc[mask, 'order_date'] + self.wait_days_median

        # Fix missing by setting median of waiting days
        mask = dataset['delivery_date'].isna()
        dataset.loc[mask, 'delivery_date'] = dataset.loc[mask, 'order_date'] + self.wait_days_median

        # Date feature engineering 
        dataset = self.add_date_features(dataset, 'delivery_date')

        # Waiting days
        dataset['waiting_days'] = (dataset['delivery_date'] - dataset['order_date']).dt.days

        # Info
        DataManager.describe_data(dataset, info=self.info)
        return dataset

    def att_user_birth(self, dataset):
        """
        Feature engineering of attribute 'user dob'
        """
        # Fix missing - Calculate median of age on clean training dataset 
        if self.training:
            clean_dataset = dataset[dataset['user_dob'].notna()]
            diff = clean_dataset['order_date'] - clean_dataset['user_dob']
            self.age_median = diff.median()
            self.age_mean = diff.mean()

        # Set calculated median
        mask = dataset['user_dob'].isna()
        dataset.loc[mask, 'user_dob'] = dataset.loc[mask, 'order_date'] - self.age_median

        # Fix invalid - Set calculated mean of age
        mask = (dataset['user_dob'].dt.year < Constants.USER_BIRTH_LOWEST) | (dataset['user_dob'].dt.year > Constants.USER_BIRTH_HIGHEST)
        dataset.loc[mask, 'user_dob'] = dataset.loc[mask, 'order_date'] - self.age_mean

        # Date feature engineering 
        dataset = self.add_date_features(dataset, 'user_dob')

        # User age
        dataset['user_age'] = (dataset['order_date'] - dataset['user_dob']).dt.days // 365

        # Same month order and bday
        dataset['same_month_bday'] = (dataset['order_date'].dt.month == dataset['user_dob'].dt.month).astype(int)

        # Same month delivery and bday
        dataset['same_month_bday_d'] = (dataset['delivery_date'].dt.month == dataset['user_dob'].dt.month).astype(int)

        # Info
        DataManager.describe_data(dataset, info=self.info)
        return dataset

    def att_user_reg_date(self, dataset):
        """
        Feature engineering of attribute 'user reg date'
        """
        # Fix invalid by setting median of waiting days
        dataset['invalid_user_reg_dates'] = dataset['user_reg_date'] > dataset['order_date']

        # Set calculated median to invalid delivery dates
        mask = dataset['invalid_user_reg_dates']
        dataset.loc[mask, 'user_reg_date'] = dataset.loc[mask, 'order_date']

        # Date feature engineering 
        dataset = self.add_date_features(dataset, 'user_reg_date')

        # User duration from order
        dataset['user_duration'] = (dataset['order_date'] - dataset['user_reg_date']).dt.days

        # User duration from delivery
        dataset['user_duration_d'] = (dataset['delivery_date'] - dataset['user_reg_date']).dt.days

        # Info
        DataManager.describe_data(dataset, info=self.info)
        return dataset

    # Categorical

    def att_item_color(self, dataset):
        """
        Feature engineering of attribute 'item color'
        """
        # Fix Missing by setting the mode 
        if self.training:
            self.item_color_counts = dataset['item_color'].value_counts()

        mask = dataset['item_color'] == '?'
        dataset.loc[mask, 'item_color'] = self.item_color_counts.index[0]

        # Feature enginering
        popular_colors = self.item_color_counts[:5]
        dataset['popular_color'] = dataset['item_color'].apply(lambda color: int(color in popular_colors))

        # Info
        DataManager.describe_data(dataset, info=self.info)
        return dataset

    def att_item_size(self, dataset):
        """
        Feature engineering of attribute 'item size'
        """
        if self.training:
            self.item_size_counts = dataset['item_size'].value_counts()

        # Feature enginering
        popular_sizes = self.item_size_counts[:5]
        dataset['popular_size'] = dataset['item_size'].apply(lambda size: int(size in popular_sizes))

        # Info
        DataManager.describe_data(dataset, info=self.info)
        return dataset

    def att_user_title(self, dataset):
        """
        Feature engineering of attribute 'user title'
        """
        # Fix Missing by setting the mode 
        if self.training:
            self.title_mode = dataset['user_title'].mode()[0]

        mask = dataset['user_title'] == 'not reported'
        dataset.loc[mask, 'user_title'] = self.title_mode

        # Info
        DataManager.describe_data(dataset, info=self.info)
        return dataset

    # Main

    def read_dataset(self, path):
        """
        Read dataset
        """
        dataset = pd.read_csv(path, header=0)
        return dataset

    def set_types(self, dataset):
        """
        Set types
        """
        # Datatime type
        for col in Constants.DATE_COLUMNS:
            dataset[col] = pd.to_datetime(dataset[col], format='%Y-%m-%d', errors='raise')

        # Int type
        dataset[Constants.CLASS_LABEL] = dataset[Constants.CLASS_LABEL].astype(int)

        return dataset

    def split_dataset(self, dataset):
        """
        Split dataset into X and y
        """
        # Split
        X_data = dataset.iloc[:, :-1]
        y_data = dataset.iloc[:, -1:]
        
        # Info
        print('\n\nTraining') if self.training else print('\n\nTesting')
        print(f'Dataset set: {len(y_data)}')
        print(f'\nDistribution:\n{y_data.value_counts().sort_index()}')

        # Ravel
        y_data = y_data.values.ravel()

        return X_data, y_data

    def remove_columns(self, dataset, columns_drop):
        """
        Remove columns
        """
        # Remove columns
        dataset = dataset.drop(columns_drop, axis=1)
        return dataset

    def clean_dataset(self, dataset):
        """
        Clean given raw dataset
        """
        # Remove columns
        columns_drop = ['order_item_id']
        dataset = self.remove_columns(dataset, columns_drop)

        # Order date
        dataset = self.att_order_date(dataset)

        # Delivery date
        dataset = self.att_delivery_date(dataset)

        # User birth
        dataset = self.att_user_birth(dataset)

        # User reg date
        dataset = self.att_user_reg_date(dataset)

        # User title
        dataset = self.att_user_title(dataset)

        # Item size
        dataset = self.att_item_size(dataset)

        # Item color
        dataset = self.att_item_color(dataset)

        # Info
        DataManager.describe_data(dataset, info=self.info)

        return dataset

    def remove_noises(self, dataset):
        """
        Remove noises/outliers with unsupervised learning method Local Outlier Factor
        """
        # Remove duplicates
        if self.info:
            dupl = dataset.duplicated().value_counts()
            print('\nDuplicates:\n', dupl)
        dataset = dataset.drop_duplicates(keep='first')

        # Get input
        X_data = dataset.iloc[:, :-1]
        X_data = UtilsManager.conv_to_numerical(X_data, Constants.CATEG_COLUMNS)
        X_data = UtilsManager.conv_to_numerical(X_data, Constants.ID_COLUMNS)

        # Create detector
        detector = LOF(n_neighbors=2, contamination=0.01, n_jobs=-1)
        #detector = HBOS(n_bins=10, alpha=0.1, tol=0.5, contamination=0.02)

        # Detect outliers
        detector.fit(X_data)
        X_data['outlier'] = detector.labels_

        # Remove it
        subset = X_data['outlier'] == 1
        indices = dataset[subset].index
        dataset = dataset.drop(indices, axis=0)
        
        # Info
        DataManager.describe_data(dataset, info=self.info, duplicates=True, visual=True)
        return dataset

    def filter_dataset(self, dataset):
        """
        Return new dataset (copy) with filter columns and/or rows
        """
        # Filter columns
        columns = [ 
            'item_id', #                          int64     Numerical
            'item_size', #                       object   Categorical  
            'item_color', #                      object   Categorical  
            'brand_id', #                         int64     Numerical
            'item_price', #                     float64     Numerical
            'user_id', #                          int64     Numerical  
            'user_title', #                      object   Categorical  
            'user_state', #                      object   Categorical 
            'order_date_month', #                 int64     Numerical 
            'order_date_day', #                   int64     Numerical 
            'order_date_dayweek', #               int64     Numerical  
            'order_date_quarter', #               int64     Numerical     
            'delivery_date_month', #              int64     Numerical  
            'delivery_date_day', #                int64     Numerical  
            'delivery_date_dayweek', #            int64     Numerical  
            'delivery_date_quarter', #            int64     Numerical
            'waiting_days', #                     int64     Numerical
            'user_dob_month', #                   int64     Numerical 
            'user_dob_day', #                     int64     Numerical  
            'user_dob_dayweek', #                 int64     Numerical
            'user_dob_quarter', #                 int64     Numerical
            'user_age', #                         int64     Numerical
            'same_month_bday', #                   int64     Numerical
            'same_month_bday_d', #                int64     Numerical 
            'user_reg_date_month', #              int64     Numerical 
            'user_reg_date_day', #                int64     Numerical 
            'user_reg_date_dayweek', #            int64     Numerical  
            'user_reg_date_quarter', #            int64     Numerical   
            'user_duration', #                    int64     Numerical
            'user_duration_d', #                  int64     Numerical
            'popular_size', #                     int64     Numerical
            'popular_color', #                    int64     Numerical
            Constants.CLASS_LABEL #               int64     Numerical
        ]
        dataset = dataset[columns]

        # Info
        DataManager.describe_data(dataset, info=self.info)
        return dataset

    def prepare_dataset(self, dataset):
        """
        Prepare given dataset
        """
        # Info
        DataManager.describe_data(dataset, info=self.info, duplicates=True, visual=True)

        # Types
        dataset = self.set_types(dataset)

        # Clean
        dataset = self.clean_dataset(dataset)

        # Fliter
        dataset = self.filter_dataset(dataset)
        
        # Remove noises/outliers
        if self.training:
            dataset = self.remove_noises(dataset)

        # Sample
        if self.sample is not None:
            dataset = dataset.sample(frac=self.sample, replace=False, random_state=Constants.RANDOM_STATE).reset_index(drop=True)

        # Transform Log
        dataset[Constants.LOG_COLUMNS] = np.log(1 + dataset[Constants.LOG_COLUMNS])

        # Visualization
        if self.visual:
            self.mngrplot.visualize_data(dataset)

        # Split
        X_data, y_data = self.split_dataset(dataset)
        return X_data, y_data

    def setup_datasets(self):
        """
        Setup clean datasets
        """
        # Train
        self.training = True
        train_dataset = self.read_dataset(path=Constants.TRAIN_DATA_PATH)
        self.X_train, self.y_train = self.prepare_dataset(train_dataset)
        
        # Test
        self.training = False
        test_dataset = self.read_dataset(path=Constants.TEST_DATA_PATH)
        self.X_test, self.y_test = self.prepare_dataset(test_dataset)
        return


if __name__ == "__main__":

    datamngr = DataManager(
        info=True, 
        visual=True,
        sample=None)

    datamngr.setup_datasets()