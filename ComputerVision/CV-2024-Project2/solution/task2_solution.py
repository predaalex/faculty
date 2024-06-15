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


def get_paths(root_path, extension):
    paths = os.listdir(root_path)
    return_paths = []
    for path in paths:
        if path.endswith(extension):
            return_paths.append(path)
    return_paths.sort()
    return return_paths


def get_last_frame(video_path):
    print(video_path)
    video = cv.VideoCapture(video_path)
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video.set(cv.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = video.read()
    video.release()
    return frame



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
        return predicted_class, outputs.item()



parking_spots_coords = [
    np.array([(1588, 781), (1542, 1011), (1751, 1045), (1776, 853)], dtype=np.int32),  # Spot 1
    np.array([(1456, 720), (1374, 928), (1570, 1045), (1618, 790)], dtype=np.int32),  # Spot 2
    np.array([(1344, 637), (1249, 832), (1408, 941), (1485, 730)], dtype=np.int32),  # Spot 3
    np.array([(1243, 582), (1139, 759), (1279, 851), (1381, 644)], dtype=np.int32),  # Spot 4
    np.array([(1173, 532), (1069, 685), (1166, 776), (1269, 590)], dtype=np.int32),  # Spot 5
    np.array([(1109, 480), (974, 632), (1084, 706), (1196, 540)], dtype=np.int32),  # Spot 6
    np.array([(1045, 448), (918, 578), (1014, 648), (1117, 490)], dtype=np.int32),  # Spot 7
    np.array([(971, 446), (864, 534), (942, 593), (1034, 487)], dtype=np.int32),  # Spot 8
    np.array([(918, 415), (817, 495), (881, 544), (982, 458)], dtype=np.int32),  # Spot 9
    np.array([(874, 385), (771, 445), (839, 506), (936, 428)], dtype=np.int32),  # Spot 10
]

test_data_path = "../train/Task2/"
solution_dir_path = './tmp/Task2/'
model_path = 'model.pth'
video_paths = get_paths(test_data_path, extension='mp4')
text_paths = get_paths(test_data_path, extension='txt')
print(video_paths)
print(text_paths)


transformer = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
])

model = models.resnet34()
num_classes = 1
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, num_classes),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

wrong_predictions = []
for video_path in video_paths:
    file = open(solution_dir_path + video_path.split('.')[0] + '.txt', 'w+')

    img = get_last_frame(test_data_path + video_path)
    writer_string = ""
    for parking_spot_idx in range(10):

        crop_img_cv = crop_polygon(img, parking_spots_coords[parking_spot_idx])
        cv.imwrite("tmp.png", crop_img_cv)
        predict_label, probability = predict_image("tmp.png", model, transformer)

        writer_string += f"{str(int(predict_label))}\n"
    file.write(writer_string[:-1])

