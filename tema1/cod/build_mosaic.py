import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb

from add_pieces_mosaic import *
from parameters import *


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i
    images = []
    dir_path = params.small_images_dir
    filenames = os.listdir(dir_path)

    for image_name in filenames:
        if params.grayscale is True:
            img_current = cv.imread(dir_path + image_name, cv.IMREAD_GRAYSCALE)
        else:
            img_current = cv.imread(dir_path + image_name)
        images.append(img_current)

    images = np.array(images)


    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    params.small_images = images


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    image_height, image_width, *image_channels = params.image.shape

    aspect_ratio = image_width/image_height

    small_image_height, small_image_width, *small_image_channels = params.small_images[0].shape

    params.num_pieces_vertical = int(small_image_width * params.num_pieces_horizontal / aspect_ratio / small_image_height)
    print(params.num_pieces_vertical, params.num_pieces_horizontal)

    # redimensioneaza imaginea
    new_h = small_image_height * params.num_pieces_vertical
    new_w = small_image_width * params.num_pieces_horizontal
    print(new_h, new_w)
    params.image_resized = cv.resize(params.image, (new_w, new_h))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
