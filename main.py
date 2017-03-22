import cv2
import argparse
import pytesseract

from time import time
from PIL import Image


# slide the region of interest with determined step and size
def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# add argument parser image and for region of interest
ap = argparse.ArgumentParser()
ap.add_argument('-im', '--image', required=True, help='path to the image')
ap.add_argument('-wd', '--width', required=True, help='region of interest width')
ap.add_argument('-hg', '--height', required=True, help='region of interest height')
ap.add_argument('-st', '--step', required=True, help='region of interest step size')
args = vars(ap.parse_args())

# start timer
start_time = time()
print ('Start recognition')

# read images and get features for region of interest
image = cv2.imread(args['image'], 0)
width = int(args['width'])
height = int(args['height'])
step = int(args['step'])

results = {}

# image processing
img = cv2.medianBlur(image, 55)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

# use tesseract for each sliding
for (x, y, window) in sliding_window(image, stepSize=step, windowSize=(width, height)):
    if window.shape[0] != height or window.shape[1] != width:
        continue

    # convert image to tesseract
    converted = Image.fromarray(window)
    text = pytesseract.image_to_string(converted)

    # write results in list
    if text.isdigit() == True:
        results[text] = [x, y, width, height]
        clone = image
        cv2.rectangle(clone, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(clone, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.imwrite('result_{}'.format(args['image']), clone)
        pass

    # show results
    clone = image.copy()
    cv2.rectangle(clone, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.imshow('Recognition', clone)
    cv2.waitKey(1)

# end timer
end_time = time() - start_time
print ('Recognition ended in {} seconds'.format(round(end_time, 2)))