import cv2 as cv
import sys
import numpy as np

def showImage(window_name, img):
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.imshow(window_name, img)
    cv.resizeWindow(window_name, 1200, 800)

print(cv.__version__)

# raw_img = cv.imread("./samples/cribbage_sample.jpg")
raw_img = cv.imread("./samples/rotated_8_spades.jpg")

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
card_images = []
root_contours = [contours[idx] for idx in range(len(hierarchy[0])) if hierarchy[0][idx][3] < 0]
card_contours = [contour for contour in root_contours if contour.size >= 100]
# alternative: contours.sort(key=cv2.contourArea, reverse=true)
for contour in card_contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(gray_img, (x, y), (x + w, y + h), (200, 0, 0), 2)
    card = gray_img[y:y + h, x:x + w]
    card_images.append(card)

    rect = cv.minAreaRect(contour)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # cv.drawContours(gray_img, [box], 0, (0,0,255), 5)

    angle = rect[2]
    rows, cols = card.shape[0], card.shape[1]
    rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    card_rotated = cv.warpAffine(card, rotation_matrix, (cols, rows))

    box = cv.boxPoints(rect)
    temp_1 = np.array([box])
    temp_2 = cv.transform(temp_1, rotation_matrix)
    temp_3 = np.int0(temp_2)
    temp_4 = temp_3[0]
    pts = np.int0(cv.transform(np.array([box]), rotation_matrix))[0]
    pts[pts < 0] = 0
    cv.drawContours(card_rotated, [pts], 0, (0,0,255), 5)

    # card_rotated_cropped = card_rotated[pts[1][1]:pts[0][1],
    #                                     pts[1][0]:pts[2][0]]

    w, h = spades_img.shape[::-1]
    match_result = cv.matchTemplate(card_rotated, spades_img, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(match_result >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(card_rotated, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    showImage("Cribbage Score - Card" + str(index + 1), card)
    showImage("Cribbage Score - Card " + str(index + 1) + " (Rotated)", card_rotated)
    index += 1

showImage("Cribbage Score - Bounded Contours", gray_img)

k = cv.waitKey(0)
