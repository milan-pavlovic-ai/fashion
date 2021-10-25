
class Constants:
    """
    Constants
    """

    # Attributes
    ID_COLUMNS = ['item_id', 'brand_id', 'user_id']
    DATE_COLUMNS = ['order_date', 'delivery_date', 'user_dob', 'user_reg_date']
    CATEG_COLUMNS = ['item_size', 'item_color','user_title', 'user_state']
    LOG_COLUMNS = ['item_price', 'delivery_date_dayweek', 'waiting_days', 'user_duration', 'user_duration_d']

    # Data
    RANDOM_STATE = 21
    CLASS_LABEL = 'return'
    USER_BIRTH_LOWEST = 1910
    USER_BIRTH_HIGHEST = 2010

    # Visualization
    COLORS = ['blue', 'red']
    TSNE_LIMIT = 50
    TSNE_NUM_ITER = 500
    VIZ_SELECT_K = -50

    # Algorithms
    LOGISTIC_REGRESSION = 'LogisticRegression'
    RANDOM_FORSET = 'RandomForest'
    LIGHT_GRADIENT_BOOSTING_MACHINE = 'LightGBM'

    # Directories
    TRAIN_DATA_PATH = 'data/input/returns_known.csv'
    TEST_DATA_PATH = 'data/input/returns_unknown_full.csv'
    MODEL_DIR = 'data/output'