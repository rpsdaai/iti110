import numpy as np
import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# from tensorflow.keras.models import Model
import os
import time
import logging

import datetime as dt

import fer_eda as eda

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

def do_datasetAugment(X_train, X_test, y_train, y_test, batch_size):
    datagen = ImageDataGenerator( 
    rescale=1./255,
    rotation_range = 10,
    horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest')

    testgen = ImageDataGenerator(rescale=1./255)
    # testgen = ImageDataGenerator()

    datagen.fit(X_train)
    batch_size = 128

    train_flow = datagen.flow(X_train, y_train, batch_size=batch_size) 
    test_flow = testgen.flow(X_test, y_test, batch_size=batch_size)

    return train_flow, test_flow

def create_tb_callback(): 
    log.info('--> create_tb_callback()')
    root_logdir = os.path.join(os.curdir, "tb_logs")

    log.info('tensorboard log: ' + root_logdir)

    # use a new directory for each run
    def get_run_logdir():    
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    run_logdir = get_run_logdir()
    log.info('run_logdir = ' + run_logdir)

    tb_callback = tf.keras.callbacks.TensorBoard(run_logdir)
    # tb_callback = tf.keras.callbacks.TensorBoard('d:\\Users\\ng_a\\test\\tb_logs\\')

    return tb_callback

def create_model_checkpoint_callback():
    log.info('--> create_model_checkpoint_callback()')


    MODEL_NAME = 'densenet121-{}'.format(dt.datetime.now().strftime("%Y%m%d-%H%M%S")) +'\\'
    log.info(MODEL_NAME)

    checkpoint_dir = os.path.join(os.curdir, MODEL_NAME)
    log.info('checkpoint_dir = ' + checkpoint_dir)

    checkpoint_dir = './' + MODEL_NAME
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-{epoch}")
    # checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    log.info('checkpoint_prefix: ' + checkpoint_prefix)
    # print(checkpoint_prefix)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True,
                                                                   monitor='val_accuracy', mode='max', save_best_only=True)
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, monitor='val_accuracy', mode='max')
 
    return model_checkpoint_callback

def create_earlystopping_callback():
    log.info('--> create_earlystopping_callback()')

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    return es_callback

def build_DenseNet(input_shape, num_blocks, num_layers_per_block, num_filters, growth_rate, dropout_rate, compress_factor, eps, num_classes):
    log.info('build_DenseNet(): num_blocks = ' + str(num_blocks) + ' num_layers_per_block = ' + str(num_layers_per_block) + ' num_filters = ' + str(num_filters) + ' growth_rate = ' + str(growth_rate) + ' dropout_rate = ' + str(dropout_rate) + ' compress_factor = ' + str(compress_factor) + ' eps = ' + str(eps) + ' num_classes = ' + str(num_classes))
    def H(inputs, num_filters, dropout_rate, eps):
        log.info('H(): num_filters = ' + str(num_filters) + ' dropout_rate = ' + str(dropout_rate) + ' eps = ' + str(eps))

        x = BatchNormalization( epsilon=eps )( inputs )
        x = Activation('relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(num_filters, kernel_size=(3, 3), use_bias=False , kernel_initializer='he_normal' )(x)
        x = Dropout(rate=dropout_rate )(x)
        return x

    def transition(inputs, num_filters, compression_factor, dropout_rate, eps):
        log.info('transition(): num_filters = ' + str(num_filters) + ' compression_factor = ' + str(compression_factor) + ' dropout_rate = ' + str(dropout_rate) + ' eps = ' + str(eps))

        # compression_factor is the 'θ'
        x = BatchNormalization(epsilon=eps)(inputs)
        x = Activation('relu')(x)
        num_feature_maps = inputs.shape[1] # The value of 'm'

        x = Conv2D(np.floor(compression_factor * num_feature_maps).astype(np.int) ,
                            kernel_size=(1, 1), use_bias=False, padding='same' , kernel_initializer='he_normal' , 
                            kernel_regularizer=tf.keras.regularizers.l2( 1e-4 ))(x)
        x = Dropout(rate=dropout_rate)(x)
        
        x = AveragePooling2D(pool_size=(2, 2))(x)
        return x

    def dense_block(inputs, num_layers, num_filters, growth_rate, dropout_rate, eps):
        log.info('dense_block(): num_filters = ' + str(num_filters) + ' growth_rate = ' + str(growth_rate) + ' dropout_rate = ' + str(dropout_rate) + ' eps = ' + str(eps))

        for i in range(num_layers): # num_layers is the value of 'l'
            conv_outputs = H(inputs, num_filters, dropout_rate, eps)
            inputs = Concatenate()([conv_outputs, inputs])
            num_filters += growth_rate # To increase the number of filters for each layer.
        return inputs, num_filters

    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, kernel_size=(3, 3), use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)

    for i in range(num_blocks):
        x, num_filters = dense_block(x, num_layers_per_block , num_filters, growth_rate , dropout_rate, eps)
        x = transition(x, num_filters , compress_factor , dropout_rate, eps)

    x = GlobalAveragePooling2D()(x) 
    x = Dense(num_classes)(x)
    outputs = tf.keras.layers.Activation('softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def compile_model(model, filename):
    log.info('--> compile_model()')
    history = model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam( learning_rate=0.0001 ), metrics=['accuracy'])
    model.summary()

    # pip install pydot
    # pip install graphviz
    # plot_model(model, show_shapes=True, to_file='densenets.png')

    return history

def train_model(model, X_train, X_test, y_train, y_test, num_epochs, batch_size, callback_list):
    log.info('--> train_model(): num_epochs = ' + str(num_epochs) + ' batch_size = ' + str(batch_size))

    history = model.fit(X_train, y_train ,batch_size, num_epochs, verbose=1, 
                    validation_data=(X_test, y_test), callbacks = callback_list)
    return history
    
def train_model_data_aug(model, train_flow, test_flow, num_epochs, X_train, X_test, batch_size, callback_list):
    log.info('--> train_model(): num_epochs = ' + str(num_epochs))
    history = model.fit(train_flow, 
                        steps_per_epoch=len(X_train) / batch_size, 
                        epochs=num_epochs,  
                        verbose=2,  
                        callbacks=callback_list,
                        validation_data=test_flow,  
                        validation_steps=len(X_test) / batch_size)
    return history

def resume_training_model(model, model_weights_directory, weights_filename, X_train, X_test, y_train, y_test, num_epochs, batch_size, callback_list):
    model.load_weights(os.path.join(model_weights_directory, weights_filename))

    if not os.path.exists(model_weights_directory):  #If the directory does not exist, create it.
        os.makedirs(model_weights_directory)
    
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_weights_directory, "model_new-{epoch:02d}.h5"),
    #                                                 save_weights_only=True,
    #                                                 monitor = 'val_accuracy',
    #                                                 save_best_only = True,
    #                                                 mode='max')
    history = model.fit(X_train, y_train, batch_size, num_epochs, verbose=1, 
                        validation_data=(X_test, y_test), callbacks = callback_list)                                                      

    return history

def save_model_weights(model, path, filename):
    log.info('--> save_model()')
    cwd = os.getcwd()
    print (cwd)
    save_model_dir = os.path.join(os.curdir + '\\' + path)
    if os.path.exists(save_model_dir) == False:
        # Create directory if it does not exist
        log.info(save_model_dir + ' does not exist! Creating ...')
        os.mkdir(save_model_dir)
    else:
        log.info(save_model_dir + ' exists!!')

    # Serialize the model into a JSON file, which will save the architecture of our model
    model2Json = model.to_json()

    model_filename = save_model_dir + '\\' + filename
    with open(model_filename + '.json', 'w') as json_file:
        json_file.write(model2Json)
    
    # Serialize the weights into a HDF5 file, which will save all parameters of our model
    model.save_weights(model_filename + '.h5')

def save_model(model, directory, filename):
    model.save(os.curdir + '\\' + directory + '\\' +filename)

def evaluate_model(model, X_test, y_test):
    log.info('--> evaluate_model()')
    eval_metrics = model.evaluate(X_test, y_test)
    return eval_metrics

if __name__ == '__main__':
    num_classes = 7
    batch_size = 128
    # # num_epochs = 30
    num_epochs = 100

    X_train, y_train, X_test, y_test = eda.load_dataset('fer2013.csv', num_classes)
    # train_flow, test_flow = do_datasetAugment(X_train, X_test, y_train, y_test, batch_size)

    input_shape=(48,48,1)
    num_blocks = 3
    num_layers_per_block = 4
    growth_rate = 16
    dropout_rate = 0.4
    compress_factor = 0.5
    eps = 1.1e-5
    num_filters = 16
 
    model = build_DenseNet(input_shape, num_blocks, num_layers_per_block, num_filters, growth_rate, dropout_rate, compress_factor, eps, num_classes)
    compile_model(model, 'desnse121.png')

    # callback_list = [create_tb_callback(), create_model_checkpoint_callback(), create_earlystopping_callback()]
    callback_list = [create_tb_callback(), create_model_checkpoint_callback()]
    # train_history = train_model(model, train_flow, test_flow, num_epochs, X_train, X_test, batch_size, callback_list)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # train_model(model, X_train, X_test, y_train, y_test, num_epochs, batch_size, callback_list)

    model_weights_directory = os.curdir + '\\models\\'
    weights_filename = 'dense10012022.h5'
    resume_training_model(model, model_weights_directory, weights_filename, X_train, X_test, y_train, y_test, num_epochs, batch_size, callback_list)
    
    save_model_weights(model, 'models', 'dense10012022')
    save_model(model, 'models', 'dense10012022')

    # score = evaluate_model(model, X_test, y_test)
    # log.info(score)

    # resume_training_model(os.curdir + '\\models\\dense09012022.h5', train_flow, test_flow, num_epochs, X_train, X_test, batch_size, callback_list)
    # tf.train.load_checkpoint('D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\iti110_working\\densenet121-20220109-160244')
    # model.load_weights(os.curdir + '\\models\\dense09012022.h5')
    score = evaluate_model(model, X_test, y_test)
    log.info(score)