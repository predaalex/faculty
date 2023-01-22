import numpy as np

from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *


params: Parameters = Parameters()
# params.dim_window = 120  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
# params.dim_hog_cell = 10  # dimensiunea celulei
# params.overlap = 0.3
#
# params.has_annotations = True
# params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = False  # adauga imaginile cu fete oglindite


facial_detector: FacialDetector = FacialDetector(params)

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente
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
    positive_descriptors, negative_descriptors = facial_detector.get_descriptors(params.nume_descriptor)
    np.save(positive_descriptors_path, positive_descriptors)
    np.save(negative_descriptors_path, negative_descriptors)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_descriptors_path)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_descriptors_path)


params.number_positive_examples = len(positive_descriptors)
params.number_negative_examples = len(negative_descriptors)

# print(f"pos descriptors \n "
#       f"{positive_descriptors[0].shape}"
#       f"{positive_descriptors[1].shape}"
#       f"{positive_descriptors[2].shape}"
#       f"{positive_descriptors.shape}"
#       )
# print(f"neg descriptors \n "
#       f"{negative_descriptors[0].shape}"
#       f"{negative_descriptors[1].shape}"
#       f"{negative_descriptors[2].shape}"
#       f"{negative_descriptors.shape}"
#       )


# Pasul 4. Invatam clasificatorul liniar
training_examples = np.concatenate((np.squeeze(positive_descriptors),
                                    np.squeeze(negative_descriptors)), axis=0)
train_labels = np.concatenate((np.ones(params.number_positive_examples), np.zeros(params.number_negative_examples)))
facial_detector.train_classifier(training_examples, train_labels)

# Pasul 5. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia FacialDetector.run()
# astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
# completati codul in continuare
# TODO:  (optional)  completeaza codul in continuare


detections, scores, file_names = facial_detector.run()


facial_detector.generate_evaluare_txt(detections, scores, file_names)

if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)