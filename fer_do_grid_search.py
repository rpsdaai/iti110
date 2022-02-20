import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from scikeras.wrappers import KerasClassifier
import fer_model as mdl
import fer_eda as eda

from enum import Enum

# Ref: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
class hyperParams(Enum):
    OPTIMIZERS = 1
    DROPOUT = 2
    LEARNING_RATE = 3
    BATCHSIZE = 4
    NONE = 5

# batch_size = 128
batch_size = 64
num_epochs = 200

input_shape=(48,48,1)
num_blocks = 3
num_layers_per_block = 4
growth_rate = 16
# dropout_rate = 0.4
dropout_rate = 0.2
compress_factor = 0.5
eps = 1.1e-5
num_filters = 16
num_classes = 7

def baseline_model_opt(optimizers):
    model = mdl.build_DenseNet(input_shape, num_blocks, num_layers_per_block, num_filters, growth_rate, dropout_rate, compress_factor, eps, num_classes)
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def baseline_model_lr(lr):
    model = mdl.build_DenseNet(input_shape, num_blocks, num_layers_per_block, num_filters, growth_rate, dropout_rate, compress_factor, eps, num_classes)
    optimizer = tf.keras.optimizers.Adam( learning_rate=lr )
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def baseline_model_bs():
    model = mdl.build_DenseNet(input_shape, num_blocks, num_layers_per_block, num_filters, growth_rate, dropout_rate, compress_factor, eps, num_classes)
    # optimizer = tf.keras.optimizers.Adam( learning_rate=lr )
    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = eda.load_dataset(eda.fer_dataset, True, num_classes)

    hyperParams = hyperParams.LEARNING_RATE
    if hyperParams == hyperParams.OPTIMIZERS:
        param_grid_opt = {
            'optimizers' : ['sgd', 'adam']
        }
        model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=baseline_model_opt, epochs=num_epochs, batch_size=batch_size, verbose=1)
        grid = GridSearchCV(estimator=model, param_grid=param_grid_opt)
    elif hyperParams == hyperParams.LEARNING_RATE:
        param_grid_lr = {
            'lr' : [0.001, 0.0001, 0.00001]
        }
        model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=baseline_model_lr, epochs=num_epochs, batch_size=batch_size, verbose=1)
        grid = GridSearchCV(estimator=model, param_grid=param_grid_lr)        
    elif hyperParams == hyperParams.BATCHSIZE:
        param_grid_bs = {
            'batch_size' : [32, 64, 128]
        }
        model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=baseline_model_bs, epochs=num_epochs, batch_size=batch_size, verbose=1)
        grid = GridSearchCV(estimator=model, param_grid=param_grid_bs)
    else:
        print ('Invalid')

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    grid_result = grid.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks = [es])

    # grid_result = model.compile(optimizer=param_grid, loss='categorical_crossentropy', metrics=['accuracy'])
    # [grid_result.best_score_,grid_result.best_params_]
    print (grid_result.best_score_)
    print (grid_result.best_params_)
