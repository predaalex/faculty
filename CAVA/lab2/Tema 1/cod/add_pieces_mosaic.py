from parameters import *
import numpy as np
import pdb
import timeit
import cv2 as cv


def get_mean_color_small_images(params: Parameters):
    N, H, W, C = params.small_images.shape

    mean_color_pieces = np.zeros((N, C))

    for i in range(N):
        img = params.small_images[i]
        for cc in range(C):
            mean_color_pieces[i, cc] = np.float32(img[:, :, cc].mean())

    return mean_color_pieces


def not_filled(indexes):
    for i in range(len(indexes)):
        for j in range(len(indexes[i])):
            if indexes[i][j][0] == -1:
                print(i, j)
                return True
    return False


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
        # print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        mean_color_pieces = get_mean_color_small_images(params)
        indexes = []
        for i in range(params.num_pieces_vertical):
            lineIndexes = []
            for j in range(params.num_pieces_horizontal):
                patch = params.image_resized[i * H: (i + 1) * H, j * W:(j + 1) * W, :].copy()
                mean_color_patch = np.mean(patch, axis=(0, 1))
                sorted_indices = get_sorted_indices(mean_color_pieces, mean_color_patch)
                dif = 0
                index = sorted_indices[dif]

                if j > 0:
                    if lineIndexes[j - 1] == index:
                        dif += 1
                        index = sorted_indices[dif]
                if i > 0:
                    if indexes[i - 1][j] == index:
                        dif += 1
                        index = sorted_indices[dif]
                if i > 0 and j > 0:
                    if indexes[i - 1][j - 1] == index:
                        dif += 1
                        index = sorted_indices[dif]

                lineIndexes.append(index)
                img_mosaic[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
            indexes.append(lineIndexes)

        # print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j) / num_pieces))

        # TODO: de implementat completarea mozaicului prin alegerea aleatorie a pozitiei unde sa fie plasate patchurile

    elif params.criterion == 'randomPatching':
        mean_color_pieces = get_mean_color_small_images(params)
        indexes = np.array([[[-1 for i in range(3)] for j in range(w)] for k in range(h)])

        # bordare stanga
        for i in range(params.num_pieces_vertical):
            j = 0
            patch = params.image_resized[i * H: (i + 1) * H, j * W:(j + 1) * W, :].copy()
            mean_color_patch = np.mean(patch, axis=(0, 1))
            sorted_indices = get_sorted_indices(mean_color_pieces, mean_color_patch)
            index = sorted_indices[0]
            img_mosaic[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
            indexes[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
        # bordare dreapta
        for i in range(params.num_pieces_vertical):
            j = params.num_pieces_horizontal - 1
            patch = params.image_resized[i * H: (i + 1) * H, j * W:(j + 1) * W, :].copy()
            mean_color_patch = np.mean(patch, axis=(0, 1))
            sorted_indices = get_sorted_indices(mean_color_pieces, mean_color_patch)
            index = sorted_indices[0]
            img_mosaic[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
            indexes[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
        # bordare sus
        i = 0
        for j in range(params.num_pieces_horizontal):
            patch = params.image_resized[i * H: (i + 1) * H, j * W:(j + 1) * W, :].copy()
            mean_color_patch = np.mean(patch, axis=(0, 1))
            sorted_indices = get_sorted_indices(mean_color_pieces, mean_color_patch)
            index = sorted_indices[0]
            img_mosaic[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
            indexes[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
        i = params.num_pieces_vertical - 1
        for j in range(params.num_pieces_horizontal):
            patch = params.image_resized[i * H: (i + 1) * H, j * W:(j + 1) * W, :].copy()
            mean_color_patch = np.mean(patch, axis=(0, 1))
            sorted_indices = get_sorted_indices(mean_color_pieces, mean_color_patch)
            index = sorted_indices[0]
            img_mosaic[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
            indexes[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()
        print(h)
        nrRandoms = 10000
        xIndex = np.random.randint(low=0, high=h - H + 1, size=nrRandoms)
        yIndex = np.random.randint(low=0, high=w - W + 1, size=nrRandoms)

        for i in range(nrRandoms):
            patch = params.image_resized[xIndex[i]:xIndex[i] + H, yIndex[i]:yIndex[i] + W, :].copy()
            mean_color_patch = np.mean(patch, axis=(0, 1))
            sorted_indices = get_sorted_indices(mean_color_pieces, mean_color_patch)
            indexPiece = sorted_indices[0]
            img_mosaic[xIndex[i]:xIndex[i] + H, yIndex[i]:yIndex[i] + W, :] = params.small_images[indexPiece].copy()
            img_mosaic[xIndex[i]:xIndex[i] + H, yIndex[i]:yIndex[i] + W, :] = params.small_images[indexPiece].copy()
        print(not_filled(indexes))
        # while not_filled(indexes):
        #     xIndex = np.random.randint(low=0, high=h - H + 1, size=1)[0]
        #     yIndex = np.random.randint(low=0, high=w - W + 1, size=1)[0]
        #
        #     patch = params.image_resized[xIndex:xIndex + H, yIndex:yIndex + W, :].copy()
        #     mean_color_patch = np.mean(patch, axis=(0, 1))
        #
        #     sorted_indices = get_sorted_indices(mean_color_pieces, mean_color_patch)
        #
        #     indexPiece = sorted_indices[0]
        #
        #     indexes[xIndex:xIndex + H, yIndex:yIndex + W, :] = params.small_images[indexPiece].copy()
        #     img_mosaic[xIndex:xIndex + H, yIndex:yIndex + W, :] = params.small_images[indexPiece].copy()

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def get_sorted_indices(mean_color_pieces, mean_color_image):
    dist = np.sum((mean_color_pieces - mean_color_image) ** 2, axis=1)

    return np.argsort(dist)


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    end_time = timeit.default_timer()
    print('running time:', (end_time - start_time), 's')
    return None


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    end_time = timeit.default_timer()
    print('running time:', (end_time - start_time), 's')
    return None
