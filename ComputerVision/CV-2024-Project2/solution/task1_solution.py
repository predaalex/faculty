import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
import torchvision.models as models
from torchvision.io import read_image
import os


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device available")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device available")
else:
    device = torch.device("cpu")
    print("CPU device available")


def get_all_paths(directory):
    images_paths = []
    query_paths = []
    gt_query_paths = []

    gt_query_paths_tmp = os.listdir(directory + "/ground-truth")
    for path in gt_query_paths_tmp:
        actual_path = directory + "ground-truth/" + path
        gt_query_paths.append(actual_path)

    file_paths = os.listdir(directory)
    for file_path in file_paths:
        if file_path.endswith(".jpg"):
            images_paths.append(directory + file_path)
        elif file_path.endswith(".txt"):
            query_paths.append(directory + file_path)

    images_paths.sort()
    query_paths.sort()
    gt_query_paths.sort()

    return images_paths, query_paths, gt_query_paths


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


def predict_image(img_path, model, transformer, score_threshold=0.8):
    img = read_image(img_path).float() / 255.0
    img = transformer(img)
    img = img.unsqueeze(0)
    img.to(device)
    with torch.no_grad():
        outputs = model(img)
        predicted_class = (outputs >= score_threshold).float().item()
        return predicted_class


def process_querry(querry_path):
    f = open(querry_path, 'r')
    nr_parking_spots = int(f.readline())
    parking_spots = []
    for i in range(nr_parking_spots):
        parking_spots.append(int(f.readline()))
    return parking_spots


def process_gt_querry(gt_querry_path):
    f = open(gt_querry_path, 'r')
    nr_parking_spots = int(f.readline())
    labels = []
    for i in range(nr_parking_spots):
        line = f.readline()
        spot_idx, label = line.split(" ")
        label = int(label)
        labels.append(label)
    return labels


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
]


data_path = "../train/Task1/"

img_paths, querry_paths, gt_querry_paths = get_all_paths(data_path)

transformer = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
])

model = models.resnet34()
num_classes = 1
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, num_classes),
    nn.Sigmoid()
)

model_path = 'model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

wrong_predictions = []

for img_path, querry_path, gt_querry_path in zip(img_paths, querry_paths, gt_querry_paths):
    original_img_cv = cv.imread(img_path)

    parking_spots_to_predict = process_querry(querry_path)
    labels = process_gt_querry(gt_querry_path)

    for spot_idx, label in zip(parking_spots_to_predict, labels):
        spot_idx -= 1

        crop_img_cv = crop_polygon(original_img_cv, parking_spots_coords[spot_idx])

        cv.imwrite("tmp.png", crop_img_cv)

        predict_label = predict_image("tmp.png", model, transformer)

        if predict_label != label:
            wrong_predictions.append(crop_img_cv)

print(len(wrong_predictions))

for wrong_prediction in wrong_predictions:
    cv.imshow("image", wrong_prediction)
    cv.waitKey(0)
cv.destroyAllWindows()
