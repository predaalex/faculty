import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
import torchvision.models as models
from torchvision.io import read_image

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device available")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device available")
else:
    device = torch.device("cpu")
    print("CPU device available")


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




# Define the polygonal coordinates for each parking spot
# Each set of points is a list of (x, y) tuples
parking_spots = [
    np.array([(1588, 781), (1542, 1011), (1751, 1045), (1776, 853)], dtype=np.int32),  # Spot 1
    np.array([(1456, 720), (1374, 928), (1570, 1045), (1618, 790)], dtype=np.int32),  # Spot 2
    np.array([(1344, 637), (1249, 832), (1408, 941), (1485, 730)], dtype=np.int32),  # Spot 3
    np.array([(1243, 582), (1139, 759), (1279, 851), (1381, 644)], dtype=np.int32),  # Spot 4
    np.array([(1173, 532), (1069, 685), (1166, 776), (1269, 590)], dtype=np.int32),  # Spot 5
    np.array([(1109, 480), (974, 632), (1084, 706), (1196, 540)], dtype=np.int32),  # Spot 6
    np.array([(1045, 448), (918, 578), (1014, 648), (1117, 490)], dtype=np.int32),  # Spot 7
    np.array([(982, 434), (871, 535), (943, 596), (1041, 486)], dtype=np.int32),  # Spot 8
    np.array([(936, 404), (827, 493), (892, 546), (989, 445)], dtype=np.int32),  # Spot 9
    np.array([(890, 377), (776, 466), (840, 518), (935, 414)], dtype=np.int32),  # Spot 10
    # np.array([(736, 506), (734, 618), (915, 653), (928, 543)], dtype=np.int32),  # Random car
    # np.array([(723, 838), (828, 1032), (1008, 1008), (898, 827)], dtype=np.int32),  # Empty car
]

# Loop through each parking spot, crop and save the image
cropped_images = []

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
model.to(device)

root_path = "../train/Task1/"
image_path = root_path + "09_1.jpg"

def predict_image(crop_img_path, score_threshold=0.8):
    img = read_image(crop_img_path).float() / 255.0
    img = transformer(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        outputs = model(img)
        print(outputs)
        predicted_class = (outputs >= score_threshold).float().item()
        return predicted_class


for i, points in enumerate(parking_spots, start=1):
    cv_image = cv.imread(image_path)

    # Crop the polygonal parking spot from the image
    crop_img = crop_polygon(cv_image, points)

    cv.imwrite("tmp.png", crop_img)
    crop_img_label = predict_image("tmp.png")

    # Save the cropped image
    cv.imshow(f"SPOT{i}|CLASS{crop_img_label}", crop_img)
    cv.imwrite("test.jpg", crop_img)
    cv.waitKey(0)
    cv.destroyWindow(f"SPOT{i}|CLASS{crop_img_label}")

cv.destroyAllWindows()
