import cv2 as cv


# In aceasta clasa vom stoca detalii legate de algoritm si de imaginea pe care este aplicat.
class Parameters:

    def __init__(self, image_path):
        self.image_path = image_path
        self.grayscale = False
        if self.grayscale is False:
            self.image = cv.imread(image_path)
        else:
            self.image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if self.image is None:
            print('%s is not valid' % image_path)
            exit(-1)

        self.image_resized = None
        self.small_images_dir = './../data/colectie/'
        self.image_type = 'png'
        self.num_pieces_horizontal = 100
        self.num_pieces_vertical = None
        self.show_small_images = False
        self.layout = 'caroiaj'
        self.criterion = 'aleator'
        self.hexagon = False
        self.small_images = None
        self.mean_color_pieces = []
        self.mask = None
        self.cifar10 = True