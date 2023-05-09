from PIL import Image
import numpy as np
from imantics import Polygons, Mask
import cv2
import random

# img = Image.open("data/v1/dev/GT/COME_Easy_1.png").convert("L")
img = Image.open("data/v1/dev/GT/COME_Easy_4580.png").convert("L")
img = np.array(img)

polygons = Mask(img).polygons()

print("len(polygons.points)", len(polygons.points))
# print(polygons.segmentation)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
print(img.shape)

for points in polygons.points:
    # print(points.shape)
    # if points.shape[0] != 6:
    #     continue
    # print(points)
    img = cv2.polylines(
        img,
        [points.reshape(-1, 1, 2)],
        isClosed=True,
        color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        thickness=5,
    )

mask = Polygons(polygons).mask(height=img.shape[0], width=img.shape[1])
mask = 255 * mask.array.astype(np.uint8)
print(mask)

cv2.imshow("img", img)
cv2.imshow("mask", mask)
cv2.waitKey()
cv2.destroyAllWindows()
