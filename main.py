import math
import cv2 as cv
import sys
import numpy as np

from cribbage import Rank, Suit

def showImage(window_name, img):
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.imshow(window_name, img)
    cv.resizeWindow(window_name, 1200, 800)

# TODO: This is broken
def isolateRankAndSuit(card_corner):
    _, threshold_img = cv.threshold(card_corner, 127, 255, 0)
    contours, hierarchy = cv.findContours(threshold_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    root_contours = [contours[idx] for idx in range(len(hierarchy[0])) if hierarchy[0][idx][3] == 0]
    if (len(root_contours) >= 2):
        x, y, w, h = cv.boundingRect(root_contours[0])
        x2, y2, w2, h2 = cv.boundingRect(root_contours[1])
        return (img[y:y+h, x:x+w], img[y2:y2+h2, x2:x2+w2])
    else:
        return (img, img)


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

raw_img = cv.imread("./samples/rotated_hand.jpg")
if raw_img is None:
    sys.exit("Could not read the image.")

def loadTemplateImage(file):
    img = cv.imread(file)
    if img is None:
        sys.exit("Could not read the image.")
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

suit_images = {
    Suit.SPADES: loadTemplateImage("./images/spades.jpg"),
    Suit.HEARTS: loadTemplateImage("./images/hearts.jpg"),
    Suit.CLUBS:  loadTemplateImage("./images/clubs.jpg"),
    Suit.DIAMONDS: loadTemplateImage("./images/diamonds.jpg")
}
rank_images = {
    Rank.ACE: loadTemplateImage("./images/ace.jpg"),
    Rank.TWO: loadTemplateImage("./images/two.jpg"),
    Rank.THREE: loadTemplateImage("./images/three.jpg"),
    Rank.FOUR: loadTemplateImage("./images/four.jpg"),
    Rank.FIVE: loadTemplateImage("./images/five.jpg"),
    Rank.SIX: loadTemplateImage("./images/six.jpg"),
    Rank.SEVEN: loadTemplateImage("./images/seven.jpg"),
    Rank.EIGHT: loadTemplateImage("./images/eight.jpg"),
    Rank.NINE: loadTemplateImage("./images/nine.jpg"),
    Rank.TEN: loadTemplateImage("./images/ten.jpg"),
    Rank.JACK: loadTemplateImage("./images/jack.jpg"),
    Rank.QUEEN: loadTemplateImage("./images/queen.jpg"),
    Rank.KING: loadTemplateImage("./images/king.jpg"),
}

def matchTemplate(threshold, image, template_image):
    w, h = template_image.shape[::-1]
    match_result = cv.matchTemplate(image, template_image, cv.TM_CCOEFF_NORMED)
    loc = np.where(match_result >= threshold)
    if (loc[0].size != 0 and loc[1].size != 0):
        for pt in zip(*loc[::-1]):
            cv.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        return True
    return False

def matchSuit(card):
    for suit in Suit:
        image = suit_images[suit]
        if (matchTemplate(0.8, card, image)):
            return suit

def matchRank(card):
    for rank in Rank:
        image = rank_images[rank]
        if (matchTemplate(0.7, card, image)):
            return rank

showImage("Cribbage Score - Original", raw_img)

img = squareImage(cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY))
showImage("Cribbage Score - Original (Grayscale + Expanded)", img)

_, threshold_img = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(threshold_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

rotated_cards = []
rotated_cropped_cards = []
corners_of_cards = []
card_suits = []
card_ranks = []
matched_cards = []
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

    rotated_box = np.int0(cv.transform(np.array([box]), rotation_matrix))[0]
    rotated_box[rotated_box < 0] = 0
    cv.drawContours(rotated_card, [rotated_box], 0, (0,0,255), 5)

    # np.sort(rotated_box)

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

    card_height, card_width = rotated_cropped_card.shape[0], rotated_cropped_card.shape[1]
    corner_of_card = rotated_cropped_card[15:int(card_height * 0.25), 15:int(card_width * 0.18)]
    corners_of_cards.append(corner_of_card)

    rank_img, suit_img = isolateRankAndSuit(corner_of_card)

    # corner_height, _ = corner_of_card.shape[0], corner_of_card[1]
    # split_point = int(corner_height * 0.6)
    card_suits.append(suit_img)
    card_ranks.append(rank_img)

    # suit = matchSuit(corner_of_card)
    # rank = matchRank(corner_of_card)
    # card = (rank, suit)
    # matched_cards.append(card)

showImage("Cribbage Score - Bounded Contours", img)

for index in range(len(rotated_cards)):
    windowName = "Cribbage Score - Card " + str(index + 1)
    showImage(windowName + " (Rotated)", rotated_cards[index])
    showImage(windowName + " (Rotated + Cropped)", rotated_cropped_cards[index])
    showImage(windowName + " (Corner)", corners_of_cards[index])
    # showImage(windowName + " (Suit)", card_suits[index])
    # showImage(windowName + " (Rank)", card_ranks[index])

for card in matched_cards:
    print(card)

k = cv.waitKey(0)
