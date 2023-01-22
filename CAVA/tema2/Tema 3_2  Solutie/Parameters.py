import os
import numpy as np

class Parameters:
    def __init__(self):
        self.base_dir = '../resources'
        self.dir_test_examples = os.path.join(self.base_dir,
                                              'validare/Validare')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.path_annotations = os.path.join(self.base_dir, 'validare/task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        # self.nume_personaje = ["andy", "louie", "ora", "tommy"]
        # self.nume_personaje = ["louie", "andy"]
        # self.nume_personaje = ["ora"]
        self.dim_window_x = None
        self.dim_window_y = None
        self.dim_hog_cell = None
        self.threshold = None
        self.nume_descriptor = None

        # pentru ora 0.55
        # self.dim_window_x = 90
        # self.dim_window_y = 130
        # self.dim_hog_cell = 10  # dimensiunea celulei
        # self.threshold = 3.75  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        # self.nume_descriptor = "ora"

        # pentru andy 0.544 -> 90 140 3.5
        # self.dim_window_x = 90
        # self.dim_window_y = 140
        # self.dim_hog_cell = 10  # dimensiunea celulei
        # self.threshold = 3.5  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        # self.nume_descriptor = "andy"

        # pentru louie 0.568
        # self.dim_window_x = 110
        # self.dim_window_y = 85
        # self.dim_hog_cell = 10  # dimensiunea celulei
        # self.threshold = 3.5  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        # self.nume_descriptor = "louie"

        # pentru tommy 0.12
        # self.dim_window_x = 120
        # self.dim_window_y = 80
        # self.dim_hog_cell = 8    # dimensiunea celulei
        # self.threshold = 5  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        # self.nume_descriptor = "tommy"

        self.crop_distance = 3
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 0  # numarul exemplelor pozitive
        self.number_negative_examples = 0  # numarul exemplelor negative
        self.detections = None  # array cu toate detectiile pe care le obtinem
        self.scores = np.array([])  # array cu toate scorurile pe care le obtinem
        self.file_names = np.array([])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori,
        self.face_clasifications = np.array([])  # array cu numele fetelor

        self.has_annotations = False
