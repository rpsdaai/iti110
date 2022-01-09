import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
import sys

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

def plot_distribution(df):
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
    plt.show()

def plot_sample_faces(df):
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

# plot_distribution(df_train)
# plot_sample_faces(df_train)
# plot_loss(history, 'loss')
# plot_accuracy(history, 'accuracy')