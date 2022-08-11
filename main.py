from curses import raw
import cv2 as cv
import sys

def showImage(window_name, img):
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.imshow(window_name, img)
    cv.resizeWindow(window_name, 1200, 800)

print(cv.__version__)

raw_img = cv.imread("./samples/cribbage_sample.jpg")

if raw_img is None:
    sys.exit("Could not read the image.")

rotated_img = cv.rotate(raw_img, cv.ROTATE_90_COUNTERCLOCKWISE)
showImage("Cribbage Score - Original", rotated_img)

gray_img = cv.cvtColor(rotated_img, cv.COLOR_BGR2GRAY)
# showImage("Cribbage Score - Gray", gray_img)

# edges_img = cv.Canny(gray_img, 100, 200)
# showImage("Cribbage Score - Edges", edges_img)

ret, thresh = cv.threshold(gray_img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(gray_img, contours, -1, (0,255,0), 3)
showImage("Cribbage Score - Contours (Threshold)", gray_img)

# contours, hierarchy = cv.findContours(edges_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(gray_img, contours, -1, (0,255,0), 3)
# showImage("Cribbage Score - Contours (Edges)", gray_img)

k = cv.waitKey(0)
