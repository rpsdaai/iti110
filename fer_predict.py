import os
import cv2
import numpy as np
import logging
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

def predict_emotion(json_file, weights_file, imageFile):
    # Load model from JSON file
    # with open(json_file, 'r') as j_file:
    # loaded_json_model = j_file.read()
    # j_file.close()
    
    loaded_model = model_from_json(open(json_file, "r").read()) 

    # Load weights and them to model
    loaded_model.load_weights(weights_file)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread(imageFile)
    
    # Convert RGB image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detects objects of different sizes in the input image. The detected objects 
    # are returned as a list of rectangles. Objects smaller than that are ignored.
    faces_detected = face_detector.detectMultiScale(gray_img, 1.18, 5)
      
    for (x, y, w, h) in faces_detected:
        # cv2.rectangle(image, start_point, end_point, color, thickness); draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = loaded_model.predict(img_pixels)
        
        # returns the position of the largest value
        max_index = int(np.argmax(predictions))
#         log.info ('max_index: ', str(max_index))
        
        # emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        emotions = [
            'Angry',
            'Disgust',
            'Fear',
            'Happy',
            'Sad',
            'Surprise',
            'Neutral'
        ]
        predicted_emotion = emotions[max_index]

        # image on which you can write the text.
        # text you want to write on image.
        # position: distance along horizontal and vertical axis from top left corner of the image.
        # font family
        # font size
        # font color
        # font stroke width
        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    resized_img = cv2.resize(img, (1024, 768))
    cv2.imshow('Facial Emotion Recognition', resized_img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def predict_emotion_from_saved_model_file(saved_model_file, imageFile):
    log.info('predict_emotion(): model_file = ' + saved_model_file + ' imaggeFile = ' + imageFile)

    loaded_model = load_model(saved_model_file)
    
    # load haarcascade to detect faces
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # load image file
    img = cv2.imread(imageFile)
    
    # Convert RGB image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detects objects of different sizes in the input image. The detected objects 
    # are returned as a list of rectangles. Objects smaller than that are ignored.
    faces_detected = face_detector.detectMultiScale(gray_img, 1.18, 5)
       
    for (x, y, w, h) in faces_detected:
        # cv2.rectangle(image, start_point, end_point, color, thickness); draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        # numpy array returned
        predictions = loaded_model.predict(img_pixels)
        
        # returns the position of the largest value
        max_index = int(np.argmax(predictions))
        # print ('max_index: ', max_index)
        # emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        emotions = [
            'Angry',
            'Disgust',
            'Fear',
            'Happy',
            'Sad',
            'Surprise',
            'Neutral'
        ]
        # emotion + probability concatented
        predicted_emotion = emotions[max_index] + ' ' + str(round(predictions[0][max_index], 2))
        # predicted_emotion = emotions[max_index]

        # image on which you can write the text.
        # text you want to write on image.
        # position: distance along horizontal and vertical axis from top left corner of the image.
        # font family
        # font size
        # font color
        # font stroke width
        # cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    resized_img = cv2.resize(img, (1024, 768))
    # resized_img = cv2.resize(img, (640, 480))
    cv2.imshow('Facial Emotion Recognition', resized_img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# Ref: https://github.com/karansjc1/emotion-detection/tree/master/without%20flask
def predict_emotion_from_video(model_file):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model(model_file)

    emotions = [
            'Angry',
            'Disgust',
            'Fear',
            'Happy',
            'Sad',
            'Surprise',
            'Neutral'
    ]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48), interpolation = cv2.INTER_AREA)

            if np.sum([roi_gray]) !=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                preds = classifier.predict(roi)[0]
                label = emotions[preds.argmax()]
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def predict_emotion_from_video_file(model_file, video_file):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model(model_file)

    emotions = [
            'Angry',
            'Disgust',
            'Fear',
            'Happy',
            'Sad',
            'Surprise',
            'Neutral'
    ]

    if os.path.exists(video_file):
        print (video_file + ' FOUND!!')
        # Read the video from specified path
        cap = cv2.VideoCapture(video_file)
    else:
        print (video_file + ' DOES NOT Exist!!!')
        
    while (cap.isOpened()):
        ret, frame = cap.read()
        labels = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48), interpolation = cv2.INTER_AREA)

            if np.sum([roi_gray]) !=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                preds = classifier.predict(roi)[0]
                label = emotions[preds.argmax()]
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #  predict_emotion('models\\xception120122.json', 'models\\xception120122.h5', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\angry.jpg') 
    #  predict_emotion('models\\xception120122.json', 'models\\xception120122.h5', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\disgust.jpg') 
    #  predict_emotion('models\\xception120122.json', 'models\\xception120122.h5', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\fear.jpg') 
    #  predict_emotion('models\\xception120122.json', 'models\\xception120122.h5', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\happy.jpg') 
    #  predict_emotion('models\\xception120122.json', 'models\\xception120122.h5', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\neutral.jpg') 
    #  predict_emotion('models\\xception120122.json', 'models\\xception120122.h5', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\sad.jpg') 
    #  predict_emotion('models\\xception120122.json', 'models\\xception120122.h5', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\surprise.jpg')
    
    # predict_emotion_from_video('models\\dense09012022')

    # predict_emotion_from_saved_model_file('models\\dense09012022', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\angry.jpg') 
    # predict_emotion_from_saved_model_file('models\\dense09012022', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\disgust.jpg') 
    # predict_emotion_from_saved_model_file('models\\dense09012022', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\fear.jpg') 
    # predict_emotion_from_saved_model_file('models\\dense09012022', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\happy.jpg') 
    # predict_emotion_from_saved_model_file('models\\dense09012022', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\neutral.jpg') 
    # predict_emotion_from_saved_model_file('models\\dense09012022', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\sad.jpg') 
    # predict_emotion_from_saved_model_file('models\\dense09012022', 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\ITI110\\test_images\\surprise.jpg')

    # now = '06022022-093427'
    # json_file = 'densenet121-' + now + '.json'
    # weights_file = 'densenet121-' + now + '.h5'
    # model_file = os.curdir + '/models/' + 'densenet121-06022022' + now
    # log.info('json_file: ' + json_file + ' weights_file: ' + weights_file)
    # model_file_h5 = 'D:\\Users\\ng_a\\My NYP SDAAI\\PDC-2\\iti110_working\\models\\densenet121-06022022\\densenet121-06022022-093427'
    # for (root,dirs,files) in os.walk('test_images', topdown=True):
    #     for file in files:
    #         # predict_emotion('models/densenet121-06022022/'+json_file, 'models/densenet121-06022022/'+weights_file, 'test_images/' + file)
    #         predict_emotion_from_saved_model_file(model_file_h5, 'test_images/' + file)
    # predict_emotion_from_video('models\\densenet121-06022022\\densenet121-06022022-093427\\')
    predict_emotion_from_video_file('models\\densenet121-06022022\\densenet121-06022022-093427\\', 'VID-20220109-WA0000.mp4')