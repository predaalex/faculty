from parameters import *
import numpy as np
import pdb
import timeit


def get_mean_color_small_images(params: Parameters):
    N, H, W, C = params.small_images.shape

    mean_color_pieces = np.zeros((N, C))

    for i in range(N):
        img = params.small_images[i]
        for cc in range(C):
            mean_color_pieces[i, cc] = np.float32(img[:, :, cc].mean())

    return mean_color_pieces


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
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                patch = params.image_resized[i * H: (i + 1) * H, j * W:(j + 1) * W, :].copy()
                mean_color_patch = np.mean(patch, axis=(0,1))
                sorted_indices = get_sorted_indices(mean_color_pieces, mean_color_patch)
                index = sorted_indices[0]
                img_mosaic[i * H: (i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[index].copy()

        # print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j) / num_pieces))
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
