from PIL import Image
import numpy as np
import skimage
import cv2
import random

# img = Image.open("data/v1/dev/GT/COME_Easy_1.png").convert("L")
img = Image.open("data/v1/dev/GT/COME_Easy_4580.png").convert("L")
img = np.array(img)

contours = skimage.measure.find_contours(img)
print(len(contours))

img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

for contour in contours:
    print(contour.dtype)
    contour[:, [0, 1]] = contour[:, [1, 0]]
    img = cv2.polylines(
        img,
        [contour.astype(np.int64).reshape(-1, 1, 2)],
        isClosed=True,
        color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        thickness=5,
    )

# https://stackoverflow.com/questions/39642680/create-mask-from-skimage-contour
# from skimage.draw import polygon
# rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
# img[rr, cc, 1] = 1


cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
