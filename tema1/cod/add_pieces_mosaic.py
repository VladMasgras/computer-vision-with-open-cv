import math

from parameters import *
import numpy as np
import pdb
import timeit


def find_best_match(img_current, params: Parameters):
    N, H, W, *C = params.small_images.shape
    distances = np.zeros(N)
    mean_color_current = img_current.mean(axis=(0, 1), dtype='float32')

    # mean_color_pieces = np.mean(params.small_images, axis=(0,1)) => 500x3

    for i in range(N):
        distances[i] = np.sqrt(np.sum((np.float64(mean_color_current) - np.float64(params.mean_color_pieces[i])) ** 2))

    return np.argsort(distances)


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, *C = params.small_images.shape
    h, w, *c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    for i in range(N):
        small_image_current = params.small_images[i]
        mean_color_piece = small_image_current.mean(axis=(0, 1), dtype=np.float64)
        params.mean_color_pieces.append(mean_color_piece)

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':

        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                img_current = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W]
                idx = find_best_match(img_current, params)[0]
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[idx]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

        '''
        for i in range(0, h - H + 1, H):
            for j in range(0, w - W + 1, W):
                img_current = params.image_resized[i:H + i, j:W + j, :]
                index = find_best_match(img_current, params)
                img_mosaic[i:H + i, j:W + j, :] = params.small_images[index, :, :, :]
        '''
    elif params.criterion == '4connect':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                img_current = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W]
                distances = find_best_match(img_current, params)
                for idx in range(len(distances)):
                    small_img = params.small_images[distances[idx]]
                    if i - 1 >= 0:
                        if np.array_equal(small_img, img_mosaic[(i - 1) * H: i * H, j * W: (j + 1) * W]):
                            continue
                    if i + 1 < h - H + 1:
                        if np.array_equal(small_img, img_mosaic[(i + 1) * H: (i + 2) * H, j * W: (j + 1) * W]):
                            continue
                    if j - 1 >= 0:
                        if np.array_equal(small_img, img_mosaic[i * H: (i + 1) * H, (j - 1) * W: j * W]):
                            continue
                    if j + 1 <= w - W + 1:
                        if np.array_equal(small_img, img_mosaic[i * H: (i + 1) * H, (j + 1) * W: (j + 2) * W]):
                            continue
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = small_img
                    break

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, *C = params.small_images.shape
    h, w, *c = params.image_resized.shape

    if params.grayscale is False:
        bigger_image = np.zeros((h + H, w + W, c[0]))
    else:
        bigger_image = np.zeros((h + H, w + W))

    bigger_image[:h, :w] = params.image_resized
    free_matrix = np.arange((h * w)).reshape(h, w)

    for i in range(N):
        small_image_current = params.small_images[i]
        mean_color_piece = small_image_current.mean(axis=(0, 1), dtype=np.float64)
        params.mean_color_pieces.append(mean_color_piece)

    while True:
        free_ = free_matrix[free_matrix > -1]

        if (len(free_) == 0):
            break
        index = np.random.randint(low=0, high=len(free_), size=1)
        row = free_[index][0] // free_matrix.shape[1]
        col = free_[index][0] % free_matrix.shape[1]

        idx = find_best_match(bigger_image[row:row + H, col:col + W:], params)[0]
        bigger_image[row:row + H, col:col + W] = params.small_images[idx]
        free_matrix[row:row + H, col:col + W] = -1

        print(len(free_))

    img_mosaic = bigger_image[:h, :w]

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, *C = params.small_images.shape
    h, w, *c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal * 2

    mask = np.zeros((H, W, *C), dtype='uint8')

    start = -1
    for i in reversed(range(H // 2)):
        start += +1
        for j in range(start, W - start):
            if params.grayscale is False:
                mask[i][j] = [1, 1, 1]
            else:
                mask[i][j] = 1

    start = -1
    for i in range(H // 2, H):
        start += +1
        for j in range(start, W - start):
            if params.grayscale is False:
                mask[i][j] = [1, 1, 1]
            else:
                mask[i][j] = 1

    params.mask = mask

    for i in range(N):
        small_image_current = params.small_images[i]
        if params.hexagon:
            small_image_current = mask * small_image_current
        mean_color_piece = small_image_current.mean(axis=(0, 1), dtype=np.float64)
        params.mean_color_pieces.append(mean_color_piece)

    if params.grayscale is False:
        bigger_image = np.zeros((h + 2 * H, w + 2 * W, c[0]))
        bigger_image2 = np.zeros((h + 2 * H, w + 2 * W, c[0]))
    else:
        bigger_image = np.zeros((h + 2 * H, w + 2 * W))
        bigger_image2 = np.zeros((h + 2 * H, w + 2 * W))

    if params.criterion == '6connect':
        first_row_start = 14
        row_index = 1
        bigger_image[:h, :w] = params.image_resized

        for i in range(first_row_start, bigger_image.shape[0] - H, H):
            col_index = 0
            for j in range(0, bigger_image.shape[1] - W, W + math.ceil(1 / 3 * W)):
                patch = bigger_image[i:i + H, j:j + W]
                patch = mask * patch
                distances = find_best_match(patch, params)
                for index in distances:
                    small_img = mask * params.small_images[index]
                    if i - H >= 0:
                        patch_to_compare = mask * bigger_image2[i - H:i, j:j + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i + H < h - H:
                        patch_to_compare = mask * bigger_image2[i + H:i + 2 * H, j:j + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i - H // 2 >= 0 and j - math.ceil(2 / 3 * W) >= 0:
                        patch_to_compare = mask * bigger_image2[i - H // 2:i + H // 2,
                                                  j - math.ceil(2 / 3 * W):j - math.ceil(2 / 3 * W) + W]

                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i - H // 2 >= 0 and j + math.ceil(2 / 3 * W) < w - W:
                        patch_to_compare = mask * bigger_image2[i - H // 2:i + H // 2,
                                                  j + math.ceil(2 / 3 * W):j + math.ceil(2 / 3 * W) + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i + H < h - H and j - math.ceil(2 / 3 * W) >= 0:
                        patch_to_compare = mask * bigger_image2[i + H // 2:i + H // 2 + H,
                                                  j - math.ceil(2 / 3 * W):j - math.ceil(2 / 3 * W) + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i + H // 2 < h - H and j + math.ceil(2 / 3 * W) < w - W:
                        patch_to_compare = mask * bigger_image2[i + H // 2:i + H // 2 + H,
                                                  j + math.ceil(2 / 3 * W):j + math.ceil(2 / 3 * W) + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue

                    bigger_image2[i:i + H, j:j + W] = (1 - mask) * bigger_image2[i:i + H, j:j + W] + mask * params.small_images[index]
                    break

                col_index += 2
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

            row_index += 2

        row_index = 0
        for i in range(0, bigger_image.shape[0] - H, H):
            col_index = 1
            for j in range(math.ceil(2 / 3 * W), bigger_image.shape[1] - W, W + math.ceil(1 / 3 * W)):
                patch = bigger_image[i:i + H, j:j + W]
                patch = mask * patch
                distances = find_best_match(patch, params)
                for index in distances:
                    small_img = mask * params.small_images[index]
                    if i - H >= 0:
                        patch_to_compare = mask * bigger_image2[i - H:i, j:j + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i + H < h - H:
                        patch_to_compare = mask * bigger_image2[i + H:i + 2 * H, j:j + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i - H // 2 >= 0 and j - math.ceil(2 / 3 * W) >= 0:
                        patch_to_compare = mask * bigger_image2[i - H // 2:i + H // 2,
                                                  j - math.ceil(2 / 3 * W):j - math.ceil(2 / 3 * W) + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i - H // 2 >= 0 and j + math.ceil(2 / 3 * W) < w - W:
                        patch_to_compare = mask * bigger_image2[i - H // 2:i + H // 2,
                                                  j + math.ceil(2 / 3 * W):j + math.ceil(2 / 3 * W) + W]

                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i + H // 2 < h - H and j - math.ceil(2 / 3 * W) >= 0:

                        patch_to_compare = mask * bigger_image2[i + H // 2:i + H // 2 + H,
                                                  j - math.ceil(2 / 3 * W):j - math.ceil(2 / 3 * W) + W]
                        if np.array_equal(small_img, patch_to_compare):
                            continue
                    if i + H // 2 < h - H and j + math.ceil(2 / 3 * W) < w - W:
                        patch_to_compare = mask * bigger_image2[i + H // 2:i + H // 2 + H,
                                                  j + math.ceil(2 / 3 * W):j + math.ceil(2 / 3 * W) + W]

                        if np.array_equal(small_img, patch_to_compare):
                            continue

                    bigger_image2[i:i + H, j:j + W] = (1 - mask) * bigger_image2[i:i + H, j:j + W] + mask * params.small_images[index]
                    break


                col_index += 2
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

            row_index += 2

        img_mosaic = bigger_image2[:h, :w]
        end_time = timeit.default_timer()
        print('Running time: %f s.' % (end_time - start_time))

        return img_mosaic

    else:
        bigger_image[:h, :w] = params.image_resized

        first_row_start = 14
        row_index = 1

        for i in range(first_row_start, bigger_image.shape[0] - H, H):
            col_index = 0
            for j in range(0, bigger_image.shape[1] - W, W + math.ceil(1 / 3 * W)):
                patch = bigger_image[i:i + H, j:j + W]
                patch = patch * mask
                index = find_best_match(patch, params)[0]
                bigger_image2[i:i + H, j:j + W] = (1 - mask) * bigger_image2[i:i + H, j:j + W] + mask * params.small_images[
                    index]

                col_index += 2
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

            row_index += 2

        row_index = 0
        for i in range(0, bigger_image.shape[0] - H, H):
            col_index = 1
            for j in range(math.ceil(2 / 3 * W), bigger_image.shape[1] - W, W + math.ceil(1 / 3 * W)):
                patch = bigger_image[i:i + H, j:j + W]
                patch = mask * patch
                index = find_best_match(patch, params)[0]
                bigger_image2[i:i + H, j:j + W] = (1 - mask) * bigger_image2[i:i + H, j:j + W] + mask * params.small_images[
                    index]
                col_index += 2
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

            row_index += 2

        img_mosaic = bigger_image2[:h, :w]

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic
