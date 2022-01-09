import numpy as np
import pandas as pd
import sys
import logging

from tensorflow.keras.utils import to_categorical

data_dir_path = ''
fer_dataset = data_dir_path+'fer2013.csv'


# Log to both console + file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler('fer.log', 'w', 'utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

# Ref: https://stackoverflow.com/questions/21137150/format-suppress-scientific-notation-from-python-pandas-aggregation-results
pd.options.display.float_format = '{:.2f}'.format

# Ref: https://colab.research.google.com/github/RodolfoFerro/PyConCo20/blob/full-code/notebooks/Deep%20Learning%20Model.ipynb
def load_dataset(filename, num_classes):
    """Utility function to load the FER2013 dataset.
    
    It returns the formated tuples (X_train, y_train) , (X_test, y_test).

    Parameters
    ==========
    filename : String
    """
    log.info('--> load_dataset(): filename = ' + filename + ' num_classes = ' + str(num_classes))

    # Load and filter in Training/not Training data:
    df = pd.read_csv(filename)
    
    training = df.loc[df['Usage'] == 'Training']
    testing = df.loc[df['Usage'] != 'Training']

    # X_train values:
    X_train = training[['pixels']].values
    X_train = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_train]

    X_train = [e.reshape((48, 48, 1)).astype('float32') for e in X_train]
    # print (type(X_train))
    X_train = np.array(X_train)
    # print (type(X_train))

    # X_train = X_train / 255.0
    
    # X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)  
    
    # X_test values:
    X_test = testing[['pixels']].values
    X_test = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_test]

    # List type
    X_test = [e.reshape((48, 48, 1)).astype('float32') for e in X_test]
    # print (type(X_test))
    
    # Convert to array
    X_test = np.array(X_test)
    # print (type(X_test))
    # X_test = X_test / 255.0

    # X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # y_train values:
    y_train = training[['emotion']].values
    # One hot encode labels
    y_train = to_categorical(y_train, num_classes)

    # y_test values
    y_test = testing[['emotion']].values
    # One hot encode labels
    y_test = to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

def do_eda(filename):
    log.info('--> do_eda(): filename = ' + filename)

    df = pd.read_csv(filename)
    log.info('HEAD \n')
    log.info(df.head())
    log.info('\n')
    log.info('COLUMNS\n')
    log.info(df.columns)
    log.info('\n')
    log.info('SHAPE\n')
    log.info(df.shape)
    log.info('\n')
    log.info('DF INFO\n')
    log.info(df.info())
    log.info('\n')
    log.info('DTYPES\n')
    log.info(df.dtypes)
    log.info('\n')
    log.info('UNIQUE EmOTIONS\n')
    log.info(df['emotion'].unique())
    log.info('\n')
    log.info('Count of different emotions\n')
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    log.info(df.loc[:, 'emotion'].value_counts())
    log.info('\n')
    log.info('Count of different usages in dataset\n')
    log.info(df['Usage'].unique())
    log.info('\n')

# Explore the dataset to get general feel
if __name__ == '__main__':
    do_eda(fer_dataset)
