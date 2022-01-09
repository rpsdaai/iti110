import cv2
import numpy as np
import logging
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

def predict_emotion(json_file, weights_file, imageFile):
    log.info('predict_emotion(): jsonfile = ' + json_file + ' weights_file = ' + weights_file + ' imaggeFile = ' + imageFile)
    
    # Load model from JSON file
#     with open(json_file, 'r') as j_file:
#         loaded_json_model = j_file.read()
#     j_file.close()
    
    loaded_model = model_from_json(open(json_file, "r").read()) 
    #loaded_model = tf.keras.models.model_from_json(loaded_model_json)

    # Load weights and them to model
    loaded_model.load_weights(weights_file)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread(imageFile)
    
    # Convert RGB image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detects objects of different sizes in the input image. The detected objects 
    # are returned as a list of rectangles. Objects smaller than that are ignored.
    faces_detected = face_detector.detectMultiScale(gray_img, 1.18, 5)

    # face_rects = face_detector.detectMultiScale(gray,
    #         scaleFactor=1.1,
    #         minNeighbors=5, 
    #         minSize=(30, 30),
    #         flags = cv.CASCADE_SCALE_IMAGE)

    # print(f'found {len(face_rects)} face(s)')

    # for rect in face_rects:
    #     cv.rectangle(img, rect, RED, 2)

    # cv.imshow('window', img)
    # cv.waitKey(0)
       
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
        print ('max_index: ', max_index)
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