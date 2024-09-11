from image import image
from region import region
import numpy as np

image_path = 'test_2.jpg'
img_obj = image(image_path, 4)
img_obj.show_image()
img_obj.show_grid_with_red_outline()

explored = np.zeros_like(img_obj.image, dtype=bool)
regions = []
for i, seed in enumerate(img_obj.seed_coordinates):
        new_region = region(img_obj.seed_coordinates[i], img_obj, reg_num=i + 1)
        regions.append(new_region)
for i in range(5):
        for current_region in regions:
                current_region.BFS_iter(1000, explored, regions)

img_obj.display_segmented_regions2()


