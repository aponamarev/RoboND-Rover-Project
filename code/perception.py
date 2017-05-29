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
def perception_step(R):
    # Perform perception steps to update R()
    # detect navigable terrain
    t_coor, t_warped, t_d, t_a = detect_navigable_terrain(R.img)
    R.terrain = t_coor
    # detect rock sample
    min_points_to_detect = 8
    s_warped, s_coord, s_d, s_a = detect_rock(R.img, min_points_to_detect)
    # cleanup sample rock warped image
    s_d_sorted_indices = np.argsort(s_d)
    s_warped_y, s_warped_x = s_warped.nonzero()
    s_warped[s_warped_y[s_d_sorted_indices[min_points_to_detect-1:]],
             s_warped_x[s_d_sorted_indices[min_points_to_detect-1:]]] = 0
    # Update R.vision_image (this will be displayed on left side of screen)
    visualization_true_value = 255
    R.vision_image[:, :, 2] = t_warped * visualization_true_value
    R.vision_image[:, :, 1] = s_warped * visualization_true_value
    obstacle = np.zeros_like(R.vision_image[:, :, 0])
    obstacle[(t_warped != t_warped.max()) | (s_warped!=s_warped.max())] = 1
    # Calculate obstacle coordinates
    o_coor = rover_coords(obstacle)
    R.vision_image[:, :, 0] = obstacle * visualization_true_value
    # Convert rover-centric pixel values to world coordinates
    rover_to_realworld_sale = 10
    navigable_x_world, navigable_y_world = pix_to_world(t_coor[0], t_coor[1],R.pos[0], R.pos[1],
                                                        R.yaw, R.worldmap.shape[0],
                                                        rover_to_realworld_sale)
    rock_x_world, rock_y_world = pix_to_world(t_coor[0], t_coor[1], R.pos[0], R.pos[1],
                                              R.yaw, R.worldmap.shape[0],
                                              rover_to_realworld_sale)
    obstacle_x_world, obstacle_y_world = pix_to_world(o_coor[1], o_coor[0], R.pos[0], R.pos[1],
                                                      R.yaw, R.worldmap.shape[0],
                                                      rover_to_realworld_sale)
    # Update R worldmap (to be displayed on right side of screen)
    R.worldmap[rock_y_world, rock_x_world, 1] += 1
    R.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    R.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    R.worldmap[R.worldmap[:, :, 1] > 0, 0] = 0
    R.worldmap = np.clip(R.worldmap, 0, 255)

    # Update R pixel distances and angles
    R.nav_dists = t_d
    R.nav_angles = t_a

    return R


def detect_rock(img, min_points_to_detect = 8):
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
    angle = None
    distance_to_rock = None
    rock_detected = rock_coord[0].size > min_points_to_detect
    if rock_detected:
        # Find distance and angle of rock pixels
        distance_to_rock, angle = to_polar_coords(*rock_coord)

    return rock_warped, rock_coord, distance_to_rock, angle

def detect_navigable_terrain(img):
    """
    detect_rock designed to detect rock and report the distance (in real world coordinates) and the angle (in degrees)
    to the rock
    """
    # 1) Apply color threshold to identify navigable and a rock
    terrain = color_thresh(img, (160, 255), 4, 0)
    # 2) Transform an image into a birds eye view
    terrain_warped = perspect_transform(terrain)
    # Remove values hallows generated by perspective transform
    terrain_warped[terrain_warped < 1] = 0
    # 3) Convert thresholded image pixel values to rover-centric coords
    terrain_coord = rover_coords(terrain_warped)
    # 4) Evaluate the distance and angle
    distance, angle = to_polar_coords(*terrain_coord)
    return terrain_coord, terrain_warped, distance, angle