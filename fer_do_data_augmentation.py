import logging
import time
import os
import tensorflow as tf
import numpy as np

import fer_eda as eda
import fer_model as mdl
import fer_test_baseline as viz
import fer_predict as predict
import fer_grid_search as gs

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

batch_size = 64
num_epochs = 200

input_shape=(48,48,1)
num_blocks = 3
num_layers_per_block = 4
growth_rate = 16
dropout_rate = 0.2
compress_factor = 0.5
eps = 1.1e-5
num_filters = 16
num_classes=7

now = time.strftime("%d%m%Y-%H%M%S")
seed = 888
tf.random.set_seed(seed)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = eda.load_dataset('fer2013.csv', scale=False, num_classes=num_classes)
    train_flow, test_flow = mdl.do_datasetAugment(X_train, X_test, y_train, y_test, batch_size)

    model = mdl.build_DenseNet(input_shape, num_blocks, num_layers_per_block, num_filters, growth_rate, dropout_rate, compress_factor, eps, num_classes)
    history = mdl.compile_model(model, lr=0.001)

    history_data_aug = model.fit(train_flow, 
                    steps_per_epoch=len(X_train) / batch_size, 
                    epochs=num_epochs,  
                    verbose=2,  
                    callbacks=[mdl.create_tb_callback(), mdl.create_model_checkpoint_callback(), mdl.create_earlystopping_callback()],
                    validation_data=test_flow,  
                    validation_steps=len(X_test) / batch_size)

    # X_test values are not scaled; must do the following steps to X_test else model evaluation will be wrong
    # ref: https://github.com/serengil/tensorflow-101/blob/master/python/facial-expression-recognition.py
    # https://www.kaggle.com/gauravsharma99/facial-emotion-recognition
    X_test /= 255
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    X_test = X_test.astype('float32')

    score = mdl.evaluate_model(model, X_test, y_test)
    log.info(score)

    viz.plot_loss(history_data_aug, 'densenet121-' + now + '_loss')   
    viz.plot_accuracy(history_data_aug, 'densenet121-' + now + '_accuracy')

    # Predict using trained model:
    y_pred = model.predict(X_test)
    y_pred = np.asarray([np.argmax(e) for e in y_pred])
    y_true = np.asarray([np.argmax(e) for e in y_test])

    viz.plot_confusion_matrix(y_true, y_pred, 'densenet121-' + now + "_cm")

    # Save model weights
    mdl.save_model_weights(model, 'models/', 'densenet121-' + now)

    # Save model
    mdl.save_model(model, 'models/', 'densenet121-' + now)

    # Do inference using the trained model
    json_file = 'densenet121-' + now + '.json'
    weights_file = 'densenet121-' + now + '.h5'
    model_file = os.curdir + '/models/' + 'densenet121-' + now
    log.info('json_file: ' + json_file + ' weights_file: ' + weights_file)
    for (root,dirs,files) in os.walk('test_images', topdown=True):
        for file in files:
            # predict_emotion('models/'+json_file, 'models/'+weights_file, 'test_images/' + file)
            predict.predict_emotion_from_saved_model_file(model_file, 'test_images/' + file)    
