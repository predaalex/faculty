import cv2 as cv
import numpy as np
import os


def get_all_paths(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if f"{directory}/{file}".endswith('.jpg'):
                paths.append(f"{directory}/{file}")
    return paths


def crop_polygon(image, points):
    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(image)

    # Create a polygon on the mask with the provided points
    cv.fillPoly(mask, [points], (255, 255, 255))

    # Apply the mask to the image
    cropped_image = cv.bitwise_and(image, mask)

    # Extract the bounding rectangle of the polygon
    x, y, w, h = cv.boundingRect(points)

    # Crop the bounding rectangle from the masked image
    cropped_image = cropped_image[y:y + h, x:x + w]

    return cropped_image


parking_spots_coords = [
    np.array([(1588, 781), (1542, 1011), (1751, 1045), (1776, 853)], dtype=np.int32),  # Spot 1
    np.array([(1456, 720), (1374, 928), (1570, 1045), (1618, 790)], dtype=np.int32),  # Spot 2
    np.array([(1344, 637), (1249, 832), (1408, 941), (1485, 730)], dtype=np.int32),  # Spot 3
    np.array([(1243, 582), (1139, 759), (1279, 851), (1381, 644)], dtype=np.int32),  # Spot 4
    np.array([(1173, 532), (1069, 685), (1166, 776), (1269, 590)], dtype=np.int32),  # Spot 5
    np.array([(1109, 480), (974, 632), (1084, 706), (1196, 540)], dtype=np.int32),  # Spot 6
    np.array([(1045, 448), (918, 578), (1014, 648), (1117, 490)], dtype=np.int32),  # Spot 7
    np.array([(913, 404), (816, 491), (886, 547), (983, 457)], dtype=np.int32),  # Spot 8
    np.array([(918, 415), (817, 495), (881, 544), (982, 458)], dtype=np.int32),  # Spot 9
    np.array([(874, 385), (771, 445), (839, 506), (936, 428)], dtype=np.int32),  # Spot 10
    np.array([(736, 506), (734, 618), (915, 653), (928, 543)], dtype=np.int32),  # Random car
    np.array([(723, 838), (828, 1032), (1008, 1008), (898, 827)], dtype=np.int32),  # Empty car
]


path = '../train/Task1'
img_paths = get_all_paths(path)

cropped_images = []

for img_path in img_paths:
    img = cv.imread(img_path)
    for i, points in enumerate(parking_spots_coords, start=1):
        crop_img = crop_polygon(img, points)
        cropped_images.append(crop_img)
    #     cv.imshow('Parking Spots', crop_img)
    #     cv.waitKey(0)
    # cv.destroyAllWindows()

print(len(cropped_images))



