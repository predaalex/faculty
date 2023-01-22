import numpy as np

from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *

params: Parameters = Parameters()
params.use_flip_images = False  # adauga imaginile cu fete oglindite

facial_detector: FacialDetector = FacialDetector(params)

all_detections = None
all_file_names = np.array([])
all_scores = np.array([])

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
nume_personaje = ["louie", "ora", "andy", "tommy"]
for nume_descriptor in nume_personaje:
    params.nume_descriptor = nume_descriptor

    if nume_descriptor == "andy":
        params.dim_window_x = 90
        params.dim_window_y = 140
        params.dim_hog_cell = 10  # dimensiunea celulei
        params.threshold = 3.5  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        params.nume_descriptor = "andy"
    elif nume_descriptor == "louie":
        params.dim_window_x = 110
        params.dim_window_y = 85
        params.dim_hog_cell = 10
        params.threshold = 3.5
        params.nume_descriptor = "louie"
    elif nume_descriptor == "ora":
        params.dim_window_x = 90
        params.dim_window_y = 130
        params.dim_hog_cell = 10  # dimensiunea celulei
        params.threshold = 3.75  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        params.nume_descriptor = "ora"
    elif nume_descriptor == "tommy":
        params.dim_window_x = 120
        params.dim_window_y = 80
        params.dim_hog_cell = 8  # dimensiunea celulei
        params.threshold = 5  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        params.nume_descriptor = "tommy"

    positive_descriptors_path = os.path.join(params.dir_save_files,
                                             'descriptoriExemplePozitive' + '_' +
                                             str(params.nume_descriptor) + '_' +
                                             str(params.dim_hog_cell) + '_' +
                                             str(params.crop_distance) + '_' +
                                             str(params.dim_window_y) + '_' +
                                             str(params.dim_window_x) + '_' +
                                             '.npy')

    negative_descriptors_path = os.path.join(params.dir_save_files,
                                             'descriptoriExempleNegative' + '_' +
                                             str(params.nume_descriptor) + '_' +
                                             str(params.dim_hog_cell) + '_' +
                                             str(params.crop_distance) + '_' +
                                             str(params.dim_window_y) + '_' +
                                             str(params.dim_window_x) + '_' +
                                             '.npy')
    nume_personaj = "louie"
    if os.path.exists(positive_descriptors_path) and os.path.exists(negative_descriptors_path):
        positive_descriptors = np.load(positive_descriptors_path)
        negative_descriptors = np.load(negative_descriptors_path)
        print(f'Am incarcat descriptorii pentru exemplele pozitive si negative pentru {params.nume_descriptor}')
    else:
        print('Construim descriptorii pentru exemplele pozitive si negative:')
        positive_descriptors, negative_descriptors = facial_detector.get_descriptors()
        np.save(positive_descriptors_path, positive_descriptors)
        np.save(negative_descriptors_path, negative_descriptors)
        print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_descriptors_path)
        print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_descriptors_path)

    params.number_positive_examples = len(positive_descriptors)
    params.number_negative_examples = len(negative_descriptors)

    # Pasul 4. Invatam clasificatorul liniar
    training_examples = np.concatenate((np.squeeze(positive_descriptors),
                                        np.squeeze(negative_descriptors)), axis=0)
    train_labels = np.concatenate((np.ones(params.number_positive_examples), np.zeros(params.number_negative_examples)))
    facial_detector.train_classifier(training_examples, train_labels)



    detections, scores, file_names = facial_detector.run()
    if all_detections is None:
        all_detections = detections
    else:
        all_detections = np.concatenate((all_detections, detections))
    all_scores = np.concatenate((all_scores, scores))
    all_file_names = np.concatenate((all_file_names, file_names))
    facial_detector.generate_evaluare_task2(detections, scores, file_names, nume_descriptor)

    # if params.has_annotations:
    #     facial_detector.eval_detections(detections, scores, file_names)
    #     show_detections_with_ground_truth(detections, scores, file_names, params)
    # else:
    #     show_detections_without_ground_truth(detections, scores, file_names, params)

# toate detectiile sortate dupa numele fisierului
n = len(all_file_names)
for i in range(n):
    for j in range(0, n - i - 1):
        if all_file_names[j] > all_file_names[j + 1]:
            all_file_names[j], all_file_names[j + 1] = all_file_names[j + 1], all_file_names[j]
            all_detections[j], all_detections[j + 1] = all_detections[j + 1], all_detections[j]
            all_scores[j], all_scores[j + 1] = all_scores[j + 1], all_scores[j]

facial_detector.generate_evaluare_task1(all_detections, all_scores, all_file_names)
