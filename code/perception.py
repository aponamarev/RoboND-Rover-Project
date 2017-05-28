import numpy as np
from matplotlib import pyplot as plt
import cv2


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, thresh=(10, 240), color=0, channel=0):
    # Create a number of available color schemes
    color_option = [None, cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2LAB, cv2.COLOR_RGB2YUV]
    assert color >= 0 & color < len(color_option), "Error: Incorrect color index"
    # Convert original image into a desired color channel
    if color_option[color] is not None:
        img_c = cv2.cvtColor(img, color_option[color])[:, :, channel]
    else:
        img_c = img[:, :, channel]

    # Create a binary image and apply threshold
    binary = np.zeros_like(img_c, dtype=np.float32)
    binary[(img_c >= thresh[0]) & (img_c <= thresh[1])] = 1
    # Return result
    return binary

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position
    #  being at thecenter bottom of the image.
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a rotation
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result
    return xpix_rotated, ypix_rotated


# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xrov_realworld, yrov_realworld, scale):
    # Apply a scaling and a translation
    xpix_translated = xrov_realworld + xpix_rot / scale
    ypix_translated = yrov_realworld + ypix_rot / scale
    # Return the result
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work


def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform


def perspect_transform(img, src=None, dst=None):

    if (src is None) | (dst is None):
        # The destination box will be 2*dst_size on each side
        # Define source and destination points for perspective transform
        dst_size = 5
        # Set a bottom offset to account for the fact that the bottom of the image
        bottom_offset = 6
        src = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
        dst = np.float32([[img.shape[1] / 2 - dst_size, img.shape[0] - bottom_offset],
                                  [img.shape[1] / 2 + dst_size, img.shape[0] - bottom_offset],
                                  [img.shape[1] / 2 + dst_size, img.shape[0] - 2 * dst_size - bottom_offset],
                                  [img.shape[1] / 2 - dst_size, img.shape[0] - 2 * dst_size - bottom_offset],
                                  ])

    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M,
                                 (img.shape[1], img.shape[0]))

    return warped


# Apply the above functions in succession and update the Rover state
# accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.binary_img
    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              ])
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    # Rover.vision_image[:,:,2] = navigable terrain color-thresholded
    # binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles

    return Rover


def detect_rock(img, rover_to_realworld_sale=10):
    """
    detect_rock designed to detect rock and report the distance (in real world coordinates) and the angle (in degrees)
    to the rock
    """
    # 1. Apply a color threshold to detect a rock in an original image
    rock = color_thresh(img, (150, 255), 3, 2)
    # 2. Transform an image into a birds eye view
    rock_warped = perspect_transform(rock)
    # Remove values hallows generated by perspective transform
    rock_warped[rock_warped < 1] = 0
    # 3. Extract rock coordinates (in rover coordinate system)
    rock_coord = rover_coords(rock_warped)
    # 3. Report if the rock is detected
    min_points_to_detect = 8
    angle = None
    distance_to_rock = None
    rock_detected = rock_coord[0].size > min_points_to_detect
    if rock_detected:
        # Find distance and angle of rock pixels
        distance_to_rock, angle = to_polar_coords(*rock_coord)
        # sort pixels by distance from low to high
        min_index = np.argsort(distance_to_rock)
        angle, distance_to_rock = angle[min_index], distance_to_rock[min_index]
        # Calculate average angle and the distance
        angle = angle[:min_points_to_detect].mean() * 180 / np.pi
        distance_to_rock = distance_to_rock[:min_points_to_detect].mean() / rover_to_realworld_sale

    return rock_detected, rock_coord, distance_to_rock, angle





def main():
    # Import pandas and read in csv file as a dataframe
    from matplotlib import pyplot as plt
    from matplotlib import image as mpimg
    import pandas as pd
    # Change this path to your data directory
    df = pd.read_csv('../test_dataset/robot_log.csv')
    # Create list of image pathnames
    csv_img_list = df["Path"].tolist()
    # Read in ground truth map and create a 3-channel image with it
    ground_truth = mpimg.imread('../calibration_images/map_bw.png')
    ground_truth_3d = np.dstack(
        (ground_truth * 0, ground_truth * 255, ground_truth * 0)).astype(np.float)

    class Databucket():
        def __init__(self):
            self.images = csv_img_list
            self.xpos = df["X_Position"].values
            self.ypos = df["Y_Position"].values
            self.yaw = df["Yaw"].values
            self.count = -1
            # This will be a running index, setting to -1 is a hack
            # because moviepy (below) seems to run one extra iteration
            self.worldmap = np.zeros((200, 200, 3)).astype(np.float)
            self.ground_truth = ground_truth_3d  # Ground truth worldmap

    # Instantiate a Databucket().. this will be a global variable/object
    # that you can refer to in the process_image() function below
    data = Databucket()

    # Define a function to pass stored images to
    # reading rover position and yaw angle from csv file
    # This function will be used by moviepy to create an output video
    def process_image(img):
        # 1) Apply color threshold to identify navigable and a rock
        terrain = color_thresh(img, (160, 255), 4, 0)
        # 2) Apply perspective transform
        terrain_warped = perspect_transform(terrain)
        # Remove values hallows generated by perspective transform
        terrain_warped[terrain_warped < 1] = 0
        # 3) Convert thresholded image pixel values to rover-centric coords
        rover_to_realworld_sale = 10
        terrain_coord = rover_coords(terrain_warped)
        rock_detected, rock_coord, rock_distance, rock_angle = detect_rock(img, rover_to_realworld_sale)
        # 4) Convert rover-centric pixel values to world coords
        terrain_world_coord = pix_to_world(terrain_coord[0], terrain_coord[1],
                                           data.xpos[data.count], data.ypos[data.count], data.yaw[data.count],
                                           data.worldmap.shape[0], rover_to_realworld_sale)

        rock_world_coord = pix_to_world(rock_coord[0], rock_coord[1],
                                           data.xpos[data.count], data.ypos[data.count], data.yaw[data.count],
                                           data.worldmap.shape[0], rover_to_realworld_sale)
        # 5) Update worldmap (to be displayed on right side of screen)
        data.worldmap[terrain_world_coord[1], terrain_world_coord[0]] = [0,0,255]
        data.worldmap[rock_world_coord[1], rock_world_coord[0]] = [255, 0, 0]
        # 6) Make a mosaic image, below is some example code
            # First create a blank image (can be whatever shape you like)
        output_image = np.zeros(
            (img.shape[0] + data.worldmap.shape[0], img.shape[1] * 2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
        output_image[0:img.shape[0], 0:img.shape[1]] = img
        # Let's create more images to add to the mosaic, first a warped image
        warped = perspect_transform(img)
        # Add the warped image in the upper right hand corner
        output_image[0:img.shape[0], img.shape[1]:] = warped
        # Overlay worldmap with ground truth map
        map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image
        output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)

        # Then putting some text over the image
        #
        msg = ["Rock was detected: {}".format(rock_detected),
               "Distance to the rock (M): {}".format(np.round(rock_distance,1) if rock_detected else "N/A"),
               "YAW(Angle) to the rock (Degrees): {}".format(np.round(rock_distance,0) if rock_detected else "N/A")]
        for i, M in enumerate(msg):
            cv2.putText(output_image, M, (20, 20 * (i+1)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
        data.count += 1  # Keep track of the index in the Databucket()

        return output_image

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from moviepy.editor import ImageSequenceClip

    # Define pathname to save the output video
    output = '../output/test_mapping.mp4'
    data = Databucket()  # Re-initialize data in case you're running this cell multiple times
    clip = ImageSequenceClip(data.images, fps=60)  # Note: output video will be sped up because
    # recording rate in simulator is fps=25
    new_clip = clip.fl_image(process_image)  # NOTE: this function expects color images!!
    new_clip.write_videofile(output, audio=False)

if __name__=="__main__":
    main()