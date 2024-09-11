from image import image
import numpy as np
from queue import Queue
import itertools
from typing import Type
import cv2
class region:
    def __init__(self, initial_seed, image_obj: image, reg_num, threshhold=30):
        self.queue = [initial_seed]
        self.reg_num = reg_num
        self.image_obj = image_obj
        self.root = initial_seed
        self.thresh = threshhold
        self.color = np.random.randint(0, 256, size=3)  # Random RGB color
        self.image_obj.getSegmentedImage()[self.root] = self.reg_num

    def BFS_iter(self, iters, explored, regions):
        # Perform one iteration of BFS
        i = 0
        while (self.queue ):
            i+= 1
            print(f'iteration {i} in region {self.reg_num}')
            current_pixel = self.queue.pop(0)
            self.image_obj.getSegmentedImage()[current_pixel] = self.reg_num

            # Add valid neighbors to the stack
            neighbours = self.getNeighbour(current_pixel[0], current_pixel[1], explored)
            self.image_obj.getSegmentedImage()[current_pixel] = self.reg_num

            x0, y0 = self.root
            for neighbor in neighbours:
                x, y = neighbor
                if (explored[x, y] == False
                    and self.distance(x, y, x0, y0)):
                    self.queue.append(neighbor)
                    explored[x, y] = True
                else:
                    # Si le pixel est déjà exploré, on regarde s'il respecte le seuil de fusion
                    # Si oui, on fusionne les deux régions
                    if self.distance(x, y, x0, y0):
                        for index, region in enumerate(regions):
                            if self.image_obj.getSegmentedImage()[x, y] == region.reg_num:
                                # On fusionne les deux régions
                                mask = (self.image_obj.getSegmentedImage() == region.reg_num)
                                self.image_obj.getSegmentedImage()[mask] = self.reg_num

                                # On supprime la région fusionnée
                                regions.remove(region)
                                break





    def in_boundaries(self, x, y):
        rows, cols = self.image_obj.getImage().shape
        return 0 <= x < rows and 0 <= y < cols

    def distance(self, x, y, x0, y0):
        return np.linalg.norm(self.image_obj.getImage()[x0, y0] - self.image_obj.getImage()[x, y]) < self.thresh
    def getNeighbour(self, x0, y0, explored):
        return [
            (x, y)
            for i, j in itertools.product((-1, 0, 1), repeat=2)
            if (i, j) != (0, 0) and self.in_boundaries(x := x0 + i, y := y0 + j)
        ]

    def display_segmented_image(self):
        segmented_image = self.image_obj.getSegmentedImage()
        segmented_image[self.root] = self.color

        # Convert image from NumPy array to uint8
        segmented_image = np.uint8(segmented_image)

        # Display the segmented image using OpenCV
        cv2.imshow('Segmented Image', segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
