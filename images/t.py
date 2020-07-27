import cv2

im1 = cv2.imread('stitched.jpg')
im1 = cv2.resize(im1, (500,375))
cv2.imwrite('stitched.jpg',im1)