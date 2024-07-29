import tensorflow as tf
from tensorflow.keras.models import load_model
import PIL
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import sys

model = load_model('final_model.h5')

def characters_in(image_path):
    img = cv2.imread(image_path)
    result = []
    margin = 20

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
   

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])


    for i, contour in enumerate(contours):
        # Get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue
    
        # Expand the bounding box to include a margin
        x -= margin
        y -= margin
        w += 2 * margin
        h += 2 * margin

        # Ensure the new coordinates are within the image boundaries
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)

        # Crop and save each character
        cropped_img = thresh[y:y + h, x:x + w]

        char_img = cv2.resize(cropped_img, (28, 28))
        char_img = np.expand_dims(char_img, axis=-1)
        char_img = char_img.mean(axis=2)
        char_img = char_img / 255
        result.append(char_img.reshape(28,28,1))
    return result

img_path = './customData/Juetguna.png'

if len(sys.argv) > 1:
    img_path = sys.argv[1]
a = characters_in(img_path)

trial = np.array([np.zeros(784)]*len(a))
for i in range(len(a)):
    image = np.flip(np.rot90(a[i],3),axis=1)
    trial[i] = image.reshape(784)
predictions = model.predict(trial)

final = ""
for i in range(len(predictions)):
    final += chr(ord('A') + np.argmax(predictions[i]))
print(f"Predicted word: {final}")
