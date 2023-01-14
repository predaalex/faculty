from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog
import time
import random


def line_info(line):
    info = line.split(" ")
    nume_imagine = info[0]
    xmin = int(info[1])
    ymin = int(info[2])
    xmax = int(info[3])
    ymax = int(info[4])
    nume_personaj = info[5][:-1]
    return nume_imagine, xmin, ymin, xmax, ymax, nume_personaj


class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None

    def get_descriptors_of_image1(self, nume_imagine_anterioara, faces_of_image, path):
        img = cv.imread(path + nume_imagine_anterioara, cv.IMREAD_GRAYSCALE)
        pos_descriptors = []
        neg_descriptors = []
        # print(nume_imagine_anterioara)
        # print(faces_of_image)
        for y in range(0, img.shape[0] - self.params.dim_window_y, self.params.crop_distance):
            for x in range(0, img.shape[1] - self.params.dim_window_x, self.params.crop_distance):
                max_iou = 0
                for face_box in faces_of_image:

                    iou = self.intersection_over_union(face_box,
                                                       (x, y, x + self.params.dim_window_x,
                                                        y + self.params.dim_window_y))
                    # print(iou)
                    if iou > max_iou:
                        max_iou = iou

                crop_img = img[y:y + self.params.dim_window_y, x:x + self.params.dim_window_x]

                features = hog(crop_img,
                               pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2),
                               # visualize=True,
                               feature_vector=True)
                # flip_features  = hog(np.fliplr(crop_img),
                #                      pixels_per_cell=(dim_hog_cell, dim_hog_cell),
                #                      cells_per_block=(2, 2),
                #                      feature_vector=True)
                # print(len(features))
                if max_iou > 0.5:
                    # cv.imshow("imghog", img_hog)
                    # cv.imshow("original", crop_img)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
                    pos_descriptors.append(features)
                    # pos_descriptors.append(flip_features)
                else:
                    # cv.imshow("original", crop_img)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
                    neg_descriptors.append(features)
                    # neg_descriptors.append(flip_features)
        # extrag random acelasi numar de descriptori negativi cu numarul de descriptori pozitivi dintr-o imagine
        pos_conter = len(pos_descriptors)
        neg_counter = len(neg_descriptors)

        random_numbers = set()
        while len(random_numbers) < pos_conter:
            random_numbers.add(random.randint(0, neg_counter - 1))

        random_numbers = list(random_numbers)

        # print(len(neg_descriptors))
        # print(len(pos_descriptors))
        aux = []
        for i in random_numbers:
            aux.append(neg_descriptors[i])
        neg_descriptors = aux

        return pos_descriptors, neg_descriptors

    def get_descriptors_of_image2(self, nume_imagine_anterioara, faces_of_image, path):
        img = cv.imread(path + nume_imagine_anterioara, cv.IMREAD_GRAYSCALE)
        pos_descriptors = []
        neg_descriptors = []

        ### creez cate un feature pentru fiecare poza redimensionata pentru descriptorii pozitivi
        for face_box in faces_of_image:
            x1, y1, x2, y2 = face_box[0], face_box[1], face_box[2], face_box[3]
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv.resize(crop_img, (self.params.dim_window_x, self.params.dim_window_y))
            # cv.imshow("crop", crop_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            features = hog(crop_img,
                           pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2),
                           # visualize=True,
                           feature_vector=True)

            # cv.imshow("imghog", img_hog)
            # cv.imshow("original", crop_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            pos_descriptors.append(features)

        pos_counter = len(pos_descriptors)

        ### iau random din aceeasi imagine, acelasi numar de descriptori negativi
        while len(neg_descriptors) < pos_counter:
            y, x = random.randint(0, img.shape[0] - self.params.dim_window_y - 1), \
                random.randint(0, img.shape[1] - self.params.dim_window_x - 1)

            max_iou = 0
            for face_box in faces_of_image:

                iou = self.intersection_over_union(face_box,
                                                   (x, y, x + self.params.dim_window_x, y + self.params.dim_window_y))
                # print(iou)
                if iou > max_iou:
                    max_iou = iou

            if max_iou > 1:
                continue
            crop_img = img[y:y + self.params.dim_window_y, x:x + self.params.dim_window_x]
            # print(f"shape = {crop_img.shape}")
            # crop_img = cv.resize(crop_img, (self.params.dim_window_x, self.params.dim_window_y))
            # print(f"shape = {crop_img.shape}")

            # cv.imshow("crop", crop_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            features = hog(crop_img,
                           pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2),
                           # visualize=True,
                           feature_vector=True)
            neg_descriptors.append(features)

        return pos_descriptors, neg_descriptors

    def get_descriptors(self, nume_personaj):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        positive_descriptors = []
        negative_descriptors = []
        nume_personaje = ["andy", "louie", "ora", "tommy"]
        # nume_personaje = ["louie"]
        for nume_personaj in nume_personaje:
            path = "../resources/antrenare/" + nume_personaj + "/"
            f = open("../resources/antrenare/" + nume_personaj + "_annotations.txt")
            lines = f.readlines()
            nume_imagine_anterioara = lines[0].split(" ")[0]
            faces_of_image = []
            time_start = time.time()
            lines.append("end0 1 2 3 4 end5\n")
            for line in lines:
                nume_imagine, xmin, ymin, xmax, ymax, nume_personaj = line_info(line)
                # if nume_personaj == "louie":
                if nume_imagine != nume_imagine_anterioara:

                    positive_descriptors_of_image, negative_descriptors_of_image = self.get_descriptors_of_image2(
                        nume_imagine_anterioara, faces_of_image, path)
                    positive_descriptors.extend(positive_descriptors_of_image)

                    negative_descriptors.extend(negative_descriptors_of_image)
                    print(f"setul de date al lui{nume_personaj}")

                    print(f"time for img{nume_imagine_anterioara} -> {time.time() - time_start}")
                    time_start = time.time()

                    nume_imagine_anterioara = nume_imagine
                    faces_of_image = [[xmin, ymin, xmax, ymax]]
                else:
                    faces_of_image.append([xmin, ymin, xmax, ymax])

            print(nume_personaj)
            print(f"imagini pozitive = {len(positive_descriptors)}")
            print(f"imagini negative = {len(negative_descriptors)}")

        self.params.number_positive_examples = len(positive_descriptors)
        self.params.number_negative_examples = len(negative_descriptors)

        positive_descriptors = np.array(positive_descriptors)
        negative_descriptors = np.array(negative_descriptors)

        print(f"shape pos = {positive_descriptors.shape}")
        print(f"shape neg = {negative_descriptors.shape}")


        return positive_descriptors, negative_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        num_negative_per_image = self.params.number_negative_examples // num_images
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            # TODO: completati codul functiei in continuare
            num_rows = img.shape[0]
            num_cols = img.shape[1]
            x = np.random.randint(low=0, high=num_cols - self.params.dim_window_x, size=num_negative_per_image)
            y = np.random.randint(low=0, high=num_rows - self.params.dim_window_y, size=num_negative_per_image)

            for idx in range(len(y)):
                patch = img[y[idx]: y[idx] + self.params.dim_window_y, x[idx]: x[idx] + self.params.dim_window_x]
                descr = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(2, 2), feature_vector=False)
                negative_descriptors.append(descr.flatten())

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        # print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:
                        # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array([])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori,
        # pentru fiecare detectie din imagine, numele imaginii va aparea in aceasta lista
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)
        descriptors_to_return = []
        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img_originala = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            # TODO: completati codul functiei in continuare
            image_scores = []
            image_detections = []
            for j in range(10, 40, 20):
                img = cv.resize(img_originala, (0, 0), fx=j / 10, fy=j / 10)
                hog_descriptors = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                      cells_per_block=(2, 2), feature_vector=False)
                num_cols = img.shape[1] // self.params.dim_hog_cell - 1
                num_rows = img.shape[0] // self.params.dim_hog_cell - 1
                num_cell_in_template_x = self.params.dim_window_x // self.params.dim_hog_cell - 1
                num_cell_in_template_y = self.params.dim_window_y // self.params.dim_hog_cell - 1

                for y in range(0, num_rows - num_cell_in_template_y):
                    for x in range(0, num_cols - num_cell_in_template_x):
                        descr = hog_descriptors[y:y + num_cell_in_template_y, x:x + num_cell_in_template_x].flatten()
                        score = np.dot(descr, w)[0] + bias
                        if score > self.params.threshold:
                            x_min = int((x / (j / 10)) * self.params.dim_hog_cell)
                            y_min = int((y / (j / 10)) * self.params.dim_hog_cell)
                            x_max = int((x / (j / 10)) * self.params.dim_hog_cell + self.params.dim_window_x / (j / 10))
                            y_max = int((y / (j / 10)) * self.params.dim_hog_cell + self.params.dim_window_y / (j / 10))
                            image_detections.append([x_min, y_min, x_max, y_max])
                            image_scores.append(score)
                            # print(image_detections)
                            # print(image_scores)
            for j in range(90, 60, -20):
                img = cv.resize(img_originala, (0, 0), fx=j / 100, fy=j / 100)
                hog_descriptors = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                      cells_per_block=(2, 2), feature_vector=False)
                num_cols = img.shape[1] // self.params.dim_hog_cell - 1
                num_rows = img.shape[0] // self.params.dim_hog_cell - 1
                num_cell_in_template_x = self.params.dim_window_x // self.params.dim_hog_cell - 1
                num_cell_in_template_y = self.params.dim_window_y // self.params.dim_hog_cell - 1

                for y in range(0, num_rows - num_cell_in_template_y):
                    for x in range(0, num_cols - num_cell_in_template_x):
                        descr = hog_descriptors[y:y + num_cell_in_template_y, x:x + num_cell_in_template_x].flatten()
                        score = np.dot(descr, w)[0] + bias
                        if score > self.params.threshold:
                            x_min = int((x / (j / 100)) * self.params.dim_hog_cell)
                            y_min = int((y / (j / 100)) * self.params.dim_hog_cell)
                            x_max = int(
                                (x / (j / 100)) * self.params.dim_hog_cell + self.params.dim_window_x / (j / 100))
                            y_max = int(
                                (y / (j / 100)) * self.params.dim_hog_cell + self.params.dim_window_y / (j / 100))
                            image_detections.append([x_min, y_min, x_max, y_max])
                            image_scores.append(score)
                            # print(image_detections)
                            # print(image_scores)

            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections),
                                                                              np.array(image_scores),
                                                                              img_originala.shape)
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
