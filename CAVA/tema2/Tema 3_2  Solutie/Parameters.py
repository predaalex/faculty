import os


class Parameters:
    def __init__(self):
        self.base_dir = '../resources'
        self.dir_pos_examples = os.path.join(self.base_dir, 'antrenare/andy')
        self.dir_neg_examples = os.path.join(self.base_dir, 'antrenare/andy')
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
        # self.dim_window = 120
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

        # pentru tommy
        self.dim_window_x = 120
        self.dim_window_y = 80
        self.dim_hog_cell = 8    # dimensiunea celulei
        self.threshold = 5  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        self.nume_descriptor = "tommy"


        self.crop_distance = 3
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 0  # numarul exemplelor pozitive
        self.number_negative_examples = 0  # numarul exemplelor negative
        self.has_annotations = True
