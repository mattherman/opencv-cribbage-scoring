import math
import cv2 as cv
import sys
import numpy as np

def showImage(window_name, img):
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.imshow(window_name, img)
    cv.resizeWindow(window_name, 1200, 800)

def squareImage(img):
    h, w = img.shape[0], img.shape[1]
    diff = abs(h - w)
    border_width = int(diff / 2)
    if (h > w):
        return cv.copyMakeBorder(img, 0, 0, border_width, border_width, cv.BORDER_CONSTANT, value=(0,0,0))
    else:
        return cv.copyMakeBorder(img, border_width, border_width, 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))

def distance(pointA, pointB):
    # sqrt((x2-x1)^2 + (y2-y1)^2)
    xDiff = pointB[0] - pointA[0]
    yDiff = pointB[1] - pointA[1]
    return math.sqrt(xDiff**2 + yDiff**2)

def calculateTransformAngle(rect):
    # The rotation transform will rotate counter-clockwise to the next 90 degree increment.
    # So if a card is tilted left like this it will end up in the wrong orientation:
    #    ______
    #    \     \        _________
    #     \     \   -> |         |
    #      \     \     |         |
    #       ------      ---------
    # To avoid that we need to identify whether the left-most point is below its
    # opposite point along the longest axis. If it is, we want to subtract 90
    # degrees from the original angle. The negative value will cause it to be rotated
    # clockwise.
        
    angle = rect[2]
    box = np.int0(cv.boxPoints(rect))
    top_point_is_furthest_left = False
    pointA = box[0]
    pointB = box[1]
    pointC = box[2]
    if (distance(pointA, pointB) > distance(pointB, pointC)):
        ax = pointA[0]
        ay = pointA[1]
        bx = pointB[0]
        by = pointB[1]
        a_is_top_and_furthest_left = (ay < by and ax < bx)
        b_is_top_and_furthest_left = (ay > by and ax > bx)
        top_point_is_furthest_left = a_is_top_and_furthest_left or b_is_top_and_furthest_left
    else:
        bx = pointB[0]
        by = pointB[1]
        cx = pointC[0]
        cy = pointC[1]
        b_is_top_and_furthest_left = (by < cy and bx < cx)
        c_is_top_and_furthest_left = (by > cy and bx > cx)
        top_point_is_furthest_left = b_is_top_and_furthest_left or c_is_top_and_furthest_left
    
    if (top_point_is_furthest_left):
        angle = angle - 90

    return angle

print(cv.__version__)

# raw_img = cv.imread("./samples/hand.jpg")
# raw_img = cv.imread("./samples/rotated_8_spades.jpg")
raw_img = cv.imread("./samples/rotated_hand.jpg")

# spades_img = cv.imread("./images/spades.jpg")
# spades_img = cv.cvtColor(spades_img, cv.COLOR_BGR2GRAY)

if raw_img is None:
    sys.exit("Could not read the image.")

showImage("Cribbage Score - Original", raw_img)

gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

img = squareImage(gray_img)
showImage("Cribbage Score - Original (Grayscale + Expanded)", img)

_, threshold_img = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(threshold_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

rotated_cards = []
rotated_cropped_cards = []
root_contours = [contours[idx] for idx in range(len(hierarchy[0])) if hierarchy[0][idx][3] < 0]
card_contours = [contour for contour in root_contours if contour.size >= 100]
# alternative: contours.sort(key=cv2.contourArea, reverse=true) ?
for contour in card_contours[:]:
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (0,0,255), 5)

    angle = calculateTransformAngle(rect)

    rows, cols = img.shape[0], img.shape[1]
    rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_card = cv.warpAffine(img, rotation_matrix, (cols, rows))
    rotated_cards.append(rotated_card)

    temp_1 = np.array([box])
    temp_2 = cv.transform(temp_1, rotation_matrix)
    temp_3 = np.int0(temp_2)
    temp_4 = temp_3[0]
    rotated_box = np.int0(cv.transform(np.array([box]), rotation_matrix))[0]
    rotated_box[rotated_box < 0] = 0
    cv.drawContours(rotated_card, [rotated_box], 0, (0,0,255), 5)

    np.sort(rotated_box)

    point_a = rotated_box[0]
    point_b = rotated_box[1]
    point_c = rotated_box[2]
    point_d = rotated_box[3]

    min_x = min(point_a[0], point_b[0], point_d[0], point_c[0])
    max_x = max(point_a[0], point_b[0], point_d[0], point_c[0])
    min_y = min(point_a[1], point_b[1], point_d[1], point_c[1])
    max_y = max(point_a[1], point_b[1], point_d[1], point_c[1])

    rotated_cropped_card = rotated_card[min_y:max_y, min_x:max_x]
    rotated_cropped_cards.append(rotated_cropped_card)
    cv.putText(rotated_cropped_card, str(angle), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv.LINE_AA)

    # w, h = spades_img.shape[::-1]
    # match_result = cv.matchTemplate(card_rotated, spades_img, cv.TM_CCOEFF_NORMED)
    # threshold = 0.8
    # loc = np.where(match_result >= threshold)
    # for pt in zip(*loc[::-1]):
    #     cv.rectangle(card_rotated, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

showImage("Cribbage Score - Bounded Contours", img)

for index, rotated_card in enumerate(rotated_cards):
    showImage("Cribbage Score - Card " + str(index + 1) + " (Rotated)", rotated_card)
for index, rotated_cropped_card in enumerate(rotated_cropped_cards):
    showImage("Cribbage Score - Card " + str(index + 1) + " (Rotated + Cropped)", rotated_cropped_card)

k = cv.waitKey(0)
