import cv2 as cv
import sys
import numpy as np

def showImage(window_name, img):
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.imshow(window_name, img)
    cv.resizeWindow(window_name, 1200, 800)

print(cv.__version__)

raw_img = cv.imread("./samples/cribbage_sample.jpg")
# raw_img = cv.imread("./samples/rotated_8_spades.jpg")

spades_img = cv.imread("./images/spades.jpg")
spades_img = cv.cvtColor(spades_img, cv.COLOR_BGR2GRAY)

if raw_img is None:
    sys.exit("Could not read the image.")

showImage("Cribbage Score - Original", raw_img)

gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
showImage("Cribbage Score - Gray", gray_img)

ret, thresh = cv.threshold(gray_img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

index = 0
root_contours = [contours[idx] for idx in range(len(hierarchy[0])) if hierarchy[0][idx][3] < 0]
card_contours = [contour for contour in root_contours if contour.size >= 100]
for contour in card_contours:
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(gray_img, [box], 0, (0,0,255), 5)

    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(gray_img, (x, y), (x + w, y + h), (200, 0, 0), 2)
    roi = gray_img[y:y + h, x:x + w]

    w, h = spades_img.shape[::-1]
    match_result = cv.matchTemplate(roi, spades_img, cv.TM_CCOEFF_NORMED)
    threshold = 0.95
    loc = np.where(match_result >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(roi, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    showImage("Cribbage Score - Card " + str(index + 1), roi)
    index += 1

showImage("Cribbage Score - Bounded Contours", gray_img)

k = cv.waitKey(0)
