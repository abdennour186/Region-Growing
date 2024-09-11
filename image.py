import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class image:
    def __init__(self, image_path, seed_grid_division=4):
        self.image_path = image_path
        self.seed_grid_division = seed_grid_division
        self.image = self.load_image()
        self.segmented_image = self.segmented_image = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.seed_coordinates = self.generate_seed_coordinates()
    def getSegmentedImage(self):
        return self.segmented_image
    def getImage(self):
        return self.image
    def load_image(self):
        try:
            image = cv2.imread(self.image_path, 0)

            if image is None:
                raise FileNotFoundError(f"Unable to load image from path: {self.image_path}")

            return image
        except Exception as e:
            raise Exception(f"Error loading image: {e}")

    def show_image(self, window_name="Image"):
        # Display the image using OpenCV
        cv2.imshow(window_name, self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def generate_seed_coordinates(self):
        rows, cols = self.image.shape

        # Divide the image into a grid
        grid_rows = np.linspace(0, rows, self.seed_grid_division + 1, dtype=int)
        grid_cols = np.linspace(0, cols, self.seed_grid_division + 1, dtype=int)
        seed_coordinates = []

        # Generate random seed coordinates within each grid
        for i in range(self.seed_grid_division):
            for j in range(self.seed_grid_division):
                x = np.random.randint(grid_rows[i], grid_rows[i + 1])
                y = np.random.randint(grid_cols[j], grid_cols[j + 1])
                seed_coordinates.append((x, y))
        return seed_coordinates

    def show_grid_with_red_outline(self, window_name="Grid with Red Outline"):
        image_with_grid = np.copy(self.image)

        # Draw red outlines around the grid borders
        rows, cols = self.image.shape
        grid_rows = np.linspace(0, rows, self.seed_grid_division + 1, dtype=int)
        grid_cols = np.linspace(0, cols, self.seed_grid_division + 1, dtype=int)

        for i in range(1, self.seed_grid_division):
            cv2.line(image_with_grid, (0, grid_rows[i]), (cols, grid_rows[i]), (0, 0, 255), 2)
            cv2.line(image_with_grid, (grid_cols[i], 0), (grid_cols[i], rows), (0, 0, 255), 2)
        inverse_coords = [(y, x) for x, y in self.seed_coordinates]
        for seed_point in inverse_coords:
            cv2.circle(image_with_grid, seed_point, 5, (0, 255, 0), -1)
        cv2.imshow(window_name, image_with_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_segmented_regions(self):
        # Create a colormap with random colors for each region
        unique_regions = np.unique(self.segmented_image)
        num_regions = len(unique_regions)
        cmap = plt.cm.get_cmap("tab20", num_regions)
        colors = cmap(np.arange(num_regions))

        # Create a colormap with black for region 0 (background)
        colors[0] = [0, 0, 0, 1]

        # Create a ListedColormap
        segmented_cmap = ListedColormap(colors)

        # Display the original image with colored regions
        plt.imshow(self.image)
        plt.imshow(self.segmented_image, cmap=segmented_cmap, alpha=0.5)  # Overlay the segmented regions
        plt.colorbar(ticks=unique_regions, label='Region Number')
        plt.show()

    def display_segmented_regions2(self):
        # Create a colormap with random colors for each region
        unique_regions = np.unique(self.segmented_image)
        num_regions = len(unique_regions)
        cmap = plt.cm.get_cmap("tab20", num_regions)
        colors = cmap(np.arange(num_regions))

        # Set black color for region 0 (background)
        colors[unique_regions == 0] = [0, 0, 0, 1]

        # Create a ListedColormap
        segmented_cmap = ListedColormap(colors)

        # Display the segmented regions without the original image
        plt.imshow(self.segmented_image, cmap=segmented_cmap)
        plt.colorbar(ticks=unique_regions, label='Region Number')
        plt.show()
