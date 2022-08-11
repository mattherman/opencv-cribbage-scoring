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
# alternative: contours.sort(key=cv2.contourArea, reverse=true)
for contour in card_contours:
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (0,0,255), 5)

    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_card = cv.warpAffine(img, rotation_matrix, (cols, rows))
    rotated_cards.append(rotated_card)

    # temp_1 = np.array([box])
    # temp_2 = cv.transform(temp_1, rotation_matrix)
    # temp_3 = np.int0(temp_2)
    # temp_4 = temp_3[0]
    pts = np.int0(cv.transform(np.array([box]), rotation_matrix))[0]
    pts[pts < 0] = 0
    cv.drawContours(rotated_card, [pts], 0, (0,0,255), 5)

    rotated_cropped_card = rotated_card[pts[1][1]:pts[0][1],
                                        pts[1][0]:pts[2][0]]
    rotated_cropped_cards.append(rotated_cropped_card)

    # w, h = spades_img.shape[::-1]
    # match_result = cv.matchTemplate(card_rotated, spades_img, cv.TM_CCOEFF_NORMED)
    # threshold = 0.8
    # loc = np.where(match_result >= threshold)
    # for pt in zip(*loc[::-1]):
    #     cv.rectangle(card_rotated, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

showImage("Cribbage Score - Bounded Contours", img)

# for index, rotated_card in enumerate(rotated_cards):
#     showImage("Cribbage Score - Card " + str(index + 1) + " (Rotated)", rotated_card)
for index, rotated_cropped_card in enumerate(rotated_cropped_cards):
    showImage("Cribbage Score - Card " + str(index + 1) + " (Rotated + Cropped)", rotated_cropped_card)

k = cv.waitKey(0)
