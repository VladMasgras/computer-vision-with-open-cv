import sys
import cv2 as cv
import numpy as np
import copy

from parameters import *
from select_path import *

import pdb


def compute_energy(img):
    """
    calculeaza energia la fiecare pixel pe baza gradientului
    :param img: imaginea initiala
    :return:E - energia
    """
    # urmati urmatorii pasi:
    # 1. transformati imagine in grayscale
    # 2. folositi filtru sobel pentru a calcula gradientul in directia X si Y
    # 3. calculati magnitudinea pentru fiecare pixel al imaginii
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sobelx = cv.Sobel(img_gray, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(img_gray, cv.CV_64F, 0, 1)

    E = np.abs(sobelx) + np.abs(sobely)

    return E


def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    E = compute_energy(img)
    new_image_E = img.copy()
    new_image_E[:, :, 0] = E.copy()
    new_image_E[:, :, 1] = E.copy()
    new_image_E[:, :, 2] = E.copy()

    for row, col in path:
        new_image_E[row, col] = color
    cv.imshow('path img', np.uint8(new_image))
    cv.imshow('path E', np.uint8(new_image_E))
    cv.waitKey(1000)


def delete_path(img, path, mask=False):
    """
    elimina drumul vertical din imagine
    :param mask: masca pentru eliminare unei portiuni din imagine
    :param img: imaginea initiala
    :path - drumul vertical
    return: updated_img - imaginea initiala din care s-a eliminat drumul vertical
    """
    if mask is True:
        updated_img = np.zeros((img.shape[0], img.shape[1] - 1))
    else:
        updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    for i in range(img.shape[0]):
        col = path[i][1]
        # copiem partea din stanga
        updated_img[i, :col] = img[i, :col].copy()
        # copiem partea din dreapta
        updated_img[i, col:] = img[i, col + 1:].copy()

    return updated_img


def decrease_width(params: Parameters, num_pixels):
    img = params.image.copy()  # copiaza imaginea originala
    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol
        if params.resize_option == 'eliminaObiect':
            E = compute_energy(img) + params.mask
        else:
            E = compute_energy(img)
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)
        if params.resize_option == 'eliminaObiect':
            params.mask = delete_path(params.mask, path, True)

    cv.destroyAllWindows()
    return img


def decrease_height(params: Parameters, num_pixels):
    img = params.image.copy()
    img_rotated = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    for i in range(num_pixels):
        print('Eliminam drumul orizontal numarul %i dintr-un total de %d.' % (i + 1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol
        if params.resize_option == 'eliminaObiect':
            E = compute_energy(img_rotated) + params.mask
        else:
            E = compute_energy(img_rotated)
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img_rotated, path, params.color_path)
        img_rotated = delete_path(img_rotated, path)
        if params.resize_option == 'eliminaObiect':
            params.mask = delete_path(params.mask, path, True)

    cv.destroyAllWindows()
    return cv.rotate(img_rotated, cv.ROTATE_90_COUNTERCLOCKWISE)


def delete_object(params: Parameters, x0, y0, w, h):
    params.mask = np.zeros((params.image.shape[0], params.image.shape[1]))
    params.mask[y0:y0 + h, x0:x0 + w] = -99999
    if w < h:
        resized_image = decrease_width(params, w)
    else:
        params.mask = cv.rotate(params.mask, cv.ROTATE_90_CLOCKWISE)
        resized_image = decrease_height(params, h)

    return resized_image


def resize_image(params: Parameters):
    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image

    elif params.resize_option == 'micsoreazaInaltime':
        resized_image = decrease_height(params, params.num_pixel_height)
        return resized_image

    elif params.resize_option == 'amplificaContinut':
        original_image = params.image
        old_width = params.image.shape[1]
        old_height = params.image.shape[0]
        print(old_width, old_height)
        new_width = int(params.image.shape[1] * params.factor_amplification)
        new_height = int(params.image.shape[0] * params.factor_amplification)
        print(new_width, new_height)
        params.image = cv.resize(params.image, (new_width, new_height))
        resized_image = decrease_width(params, new_width - old_width)
        params.image = resized_image
        resized_image = decrease_height(params, new_height - old_height)
        params.image = original_image
        print(resized_image.shape)
        return resized_image

    elif params.resize_option == 'eliminaObiect':

        resized_image = delete_object(params, params.r[0], params.r[1], params.r[2], params.r[3])

        return resized_image


    else:
        print('The option is not valid!')
        sys.exit(-1)
