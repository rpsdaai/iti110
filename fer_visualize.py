import fer_eda as eda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
import sys
import math
import tensorflow as tf

# pip install visualkeras (dependencies on aggdraw)
# pip install aggdraw 
#
# To install aggdraw on windows, download https://github.com/ubawurinna/freetype-windows-binaries
# copy the files under include directory to D:\Program Files\Java\jdk-11.0.9\include and the files
# under freetype-windows-binaries-master\release dll\win64 (freetype.lib, freetype.dll) to  D:\Program Files\Java\jdk-11.0.9\lib
#
import visualkeras
from PIL import ImageFont
from matplotlib import font_manager
from PIL import Image, ImageDraw

# Global variable
emotion_description = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

def plot_distribution(df, filename):
    log.info('--> plot_distribution()\n')
    # Ref: https://www.kaggle.com/lxyuan0420/facial-expression-recognition-using-cnn
    emotion_counts = df['emotion'].value_counts(sort=False).reset_index()
    emotion_counts.columns = ['emotion', 'number']
    emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_description)

    plt.figure(figsize=(6,6))
    
    bp = sns.barplot(emotion_counts.emotion, emotion_counts.number)
    # Ref: https://stackoverflow.com/questions/62002434/how-to-add-data-labels-to-seaborn-barplot
    for p in bp.patches:
             bp.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')
            
    # Ref: https://stackoverflow.com/questions/61368851/how-to-rotate-seaborn-barplot-x-axis-tick-labels
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    plt.title('Facial Emotion Frequency Distribution')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Emotion Types', fontsize=12)
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_sample_faces(df, filename):
    log.info('--> plot_sample_faces()\n')
    fig = plt.figure(1, (14, 14))

    k = 0
    for label in sorted(df.emotion.unique()):
        for j in range(len(df.emotion.unique())):
            px = df[df.emotion==label].pixels.iloc[k]
            px = np.array(px.split(' ')).reshape(48, 48).astype('float32')

            k += 1
            ax = plt.subplot(7, 7, k)
            ax.imshow(px, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(emotion_description[label])
            plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_loss(history, filename):
    log.info('--> plot_loss() filename = ' + filename + '\n')

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model's training loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_accuracy(history, filename):
    log.info('--> plot_accuracy(): filename = ' + filename + '\n')

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model's training accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    log.info('-->plot_confusion_matrix()\n')

    # Compute confusion matrix:
    cm = confusion_matrix(y_true, y_pred)
    cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix:
    sns.set(font_scale=1.5) 
    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(cm_normalised, annot=True, linewidths=0, square=False, 
                     cmap='gray', yticklabels=emotion_labels,
                     xticklabels=emotion_labels, vmin=0,
                     vmax=np.max(cm_normalised), fmt=".2f",
                     annot_kws={"size": 20})
    ax.set(xlabel='Predicted label', ylabel='True label')

# Plots the Deep Learning Model architecture using visual keras
def plot_dl_architecture(model, filename):
    log.info('-->plot_dl_architecture(): filename = ' + filename)

    # Install true type fonts
    # Ref: https://www.technipages.com/windows-10-how-to-install-truetype-fonts
    font = font_manager.FontProperties(family='sans-serif', weight='bold')
    file = font_manager.findfont(font)
    print(file)

    # Ref: https://stackoverflow.com/questions/66274858/choosing-a-pil-imagefont-by-font-name-rather-than-filename-and-cross-platform-f
    # Ref: https://stackoverflow.com/questions/61518366/is-possible-to-link-an-online-font-file-to-pil-imagefont-truetype
    font = ImageFont.truetype("D:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\DejaVuSans-Bold.ttf", 42, encoding="unic")
    # Ref: https://openprojectrepo.com/project/paulgavrikov-visualkeras-python-data-validation
    visualkeras.layered_view(model, legend=True, font=font, to_file=filename).show()

def plot_miss_classified(emotion, ytest_, yhat_test, X_test, model):
    emotion_text_to_labels = dict((v,k) for k,v in emotion_description.items())

    miss_happy_indices = np.where((ytest_ != yhat_test) & (ytest_ == emotion_text_to_labels[emotion]))[0]
    print(f"total {len(miss_happy_indices)} miss labels out of {len(np.where(ytest_== emotion_text_to_labels[emotion])[0])} for emotion {emotion}")

    cols = 15
    rows = math.ceil(len(miss_happy_indices) / cols)
    fig = plt.figure(1, (20, rows * 2))

    for i,idx in enumerate(miss_happy_indices):
        sample_img = X_test[idx,:,:,:]
        sample_img = sample_img.reshape(1,*sample_img.shape)
        pred = emotion_description[np.argmax(model.predict(sample_img), axis=1)[0]]

        ax = plt.subplot(rows,cols,i+1)
        ax.imshow(sample_img[0,:,:,0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"p:{pred}")  

if __name__ == '__main__':
    data_dir_path = ''
    df = pd.read_csv(data_dir_path+'fer2013.csv')

    df_train = df[df['Usage']=='Training']
    df_valid = df[df['Usage']=='PrivateTest']
    df_test = df[df['Usage']=='PublicTest']

    # Plot data distribution of FER dataset
    plot_distribution(df_train)

    # Plot a sample of the FER dataset
    plot_sample_faces(df_train)

    # plot_loss(history, 'loss')
    # plot_accuracy(history, 'accuracy')

    # font = font_manager.FontProperties(family='sans-serif', weight='bold')
    # file = font_manager.findfont(font)
    # print(file)

    # Plot misclassifications
    X_train, y_train, X_test, y_test = eda.load_dataset(eda.fer_dataset, scale=True, num_classes=7)
    loaded_model = tf.keras.models.load_model('experiments/run_10-experiment-6Feb2022/models/densenet121-06022022-093427/', compile=True)
    yhat_test = np.argmax(loaded_model.predict(X_test), axis=1)
    ytest_ = np.argmax(y_test, axis=1)

    for emotion in emotion_labels:
        plot_miss_classified(emotion, ytest_, yhat_test, X_test, loaded_model)

