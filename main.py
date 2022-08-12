import math
import cv2 as cv
import sys
import numpy as np

from cribbage import Rank, Suit

def showImage(window_name, image, size=None):
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.imshow(window_name, image)
    if size is not None:
        cv.resizeWindow(window_name, size[0], size[1])

def isolateRankAndSuit(card_corner):
    _, threshold_image = cv.threshold(card_corner, 180, 255, 0)
    contours, hierarchy = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    if (len(contours) != 0):
        root_contours = [contours[idx] for idx in range(len(hierarchy[0])) if hierarchy[0][idx][3] >= 0]
        root_contours = sorted(root_contours, key=lambda x: cv.contourArea(x), reverse=True)
        if (len(root_contours) > 1):
            x, y, w, h = cv.boundingRect(root_contours[0])
            x2, y2, w2, h2 = cv.boundingRect(root_contours[1])
            first = card_corner[y:y+h, x:x+w]
            second = card_corner[y2:y2+h2, x2:x2+w2]
            if y < y2:
                return (first, second)
            else:
                return (second, first)
    else:
        return (card_corner, card_corner)

def squareImage(image):
    h, w = image.shape[0], image.shape[1]
    diff = abs(h - w)
    border_width = int(diff / 2)
    if (h > w):
        return cv.copyMakeBorder(image, 0, 0, border_width, border_width, cv.BORDER_CONSTANT, value=(0,0,0))
    else:
        return cv.copyMakeBorder(image, border_width, border_width, 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))

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

# raw_image = cv.imread("./samples/all_cards.jpg")
raw_image = cv.imread("./samples/rotated_hand.jpg")
if raw_image is None:
    sys.exit("Could not read the image.")

def loadTemplateImage(file):
    image = cv.imread(file)
    if image is None:
        sys.exit("Could not read the image.")
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray_image

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

def matchTemplate(name, threshold, image, template_image):
    image_width, image_height = image.shape[0], image.shape[1]
    resized_template_image = cv.resize(template_image, (image_height, image_width), interpolation=cv.INTER_AREA)
    w, h = resized_template_image.shape[::-1]
    match_result = cv.matchTemplate(image, resized_template_image, cv.TM_CCOEFF_NORMED)
    loc = np.where(match_result >= threshold)
    if (loc[0].size != 0 and loc[1].size != 0):
        for pt in zip(*loc[::-1]):
            cv.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        return True
    return False

def matchSuit(card):
    for suit in Suit:
        image = suit_images[suit]
        if (matchTemplate(str(suit), 0.8, card, image)):
            return suit

def matchRank(card):
    for rank in Rank:
        image = rank_images[rank]
        if (matchTemplate(str(rank), 0.7, card, image)):
            return rank

showImage("Cribbage Score - Original", raw_image, (1200, 800))

image = squareImage(cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY))

_, threshold_image = cv.threshold(image, 127, 255, 0)
contours, hierarchy = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

rotated_cards = []
rotated_cropped_cards = []
corners_of_cards = []
card_suits = []
card_ranks = []
matched_cards = []
root_contours = [contours[idx] for idx in range(len(hierarchy[0])) if hierarchy[0][idx][3] < 0]
card_contours = sorted(root_contours, key=lambda x: cv.contourArea(x), reverse=True)
for contour in card_contours[:5]:
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(image, [box], 0, (0,255,0), 10)

    angle = calculateTransformAngle(rect)

    rows, cols = image.shape[0], image.shape[1]
    rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_card = cv.warpAffine(image, rotation_matrix, (cols, rows))
    rotated_cards.append(rotated_card)

    rotated_box = np.int0(cv.transform(np.array([box]), rotation_matrix))[0]
    rotated_box[rotated_box < 0] = 0

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
    corner_of_card = rotated_cropped_card[0:int(card_height * 0.25), 0:int(card_width * 0.18)]
    corners_of_cards.append(corner_of_card)

    rank_image, suit_image = isolateRankAndSuit(corner_of_card)

    card_suits.append(suit_image)
    card_ranks.append(rank_image)

    suit = matchSuit(suit_image)
    rank = matchRank(rank_image)
    card = (rank, suit)
    matched_cards.append(card)

showImage("Cribbage Score - Bounded Contours", image, (1200, 800))

for index in range(len(rotated_cards)):
    windowName = "Cribbage Score - Card " + str(index + 1)
    # showImage(windowName + " (Rotated)", rotated_cards[index])
    showImage(windowName + " (Rotated + Cropped)", rotated_cropped_cards[index])
    showImage(windowName + " (Corner)", corners_of_cards[index])
    # showImage(windowName + " (Suit)", card_suits[index])
    # showImage(windowName + " (Rank)", card_ranks[index])

for card in matched_cards:
    print(card)

k = cv.waitKey(0)
