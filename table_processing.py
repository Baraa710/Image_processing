import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("./images/arabicTable.png")

#Process the image using open cv
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_canny = cv.Canny(img,50, 150)
img_gray_canny = cv.Canny(img_gray,50, 150)
ret,thresh_binary = cv.threshold(img_gray_canny,127,255,cv.THRESH_BINARY)
ret,thresh_inv = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV)
ret,thresh_inv_canny = cv.threshold(img_gray_canny,127,255,cv.THRESH_BINARY_INV)
ret,thresh_trunc = cv.threshold(img_gray,127,255,cv.THRESH_TRUNC)
ret,thresh_tozero = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO)
ret,thresh_tozero_inv = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO_INV)

#store titles and images in lists
titles = ['Original Image','Gray','Canny','Gray Canny','BINARY','BINARY_INV', 'Binary INV Canny','TRUNC','TOZERO','TOZERO_INV']
images = [img, img_gray, img_canny, img_gray_canny, thresh_binary, thresh_inv, thresh_inv_canny,thresh_trunc, thresh_tozero, thresh_tozero_inv]

rows = 5
columns = 2

#assert the numbers of titles, images, rows and columns are valid
assert len(images) == len(titles), "There is not the same number of images as titles"
assert rows*columns >= len(images), "Not enough rows and columns"


fig = plt.figure(figsize=(10,7))

#loop through images and titles and add them to the plot using subplot
for i in range(len(images)):
 plt.subplot(rows,columns,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])
plt.show()
