
import cv2 as cv

query = '/train/template_0_0.jpg'
gallery = '/train/train_0.jpg'
Qu = cv.imread(query)
Ga = cv.imread(gallery)
list_of_boxes = predict_image(Ga, Qu)
