[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[grid_rock]: ./output/grid_rock.jpg
[samplerock]: ./output/samplerock.jpg
[terrain]: ./output/terrain.jpg
[warped]: ./output/warped.jpg
[warped_terrain]: ./output/warped_terrain.jpg
[process_img]: ./output/process_img.png

[video1]: ./output/test_mapping.mp4 "Video1"
[video2]: ./output/test_mapping.mp4 "Video2"

## Project: Search and Sample Return
---
**The goals / steps of this project are the following:**  

### Training / Calibration

The overall approach is described in code/Rover_Project_Test_Notebook.ipynb. 

#### Add functions to detect obstacles and samples of interest (golden rocks)

In order to identify navigable terrain, obstacles, and samples of interest I created analyze_color(img, high, low, color=0, channel=0) (utility function) that helps to try various color channels and thresholds. The function is located in code cell 6 of the ipython book.
Next, I identified the best color channels and the best thresholds for:
* navigable terrain (code cell 7): Navigable terrain is best identified in YUV color scheme on Y channel. For the threshold I used the following bounds (160, 255).
![alt text][terrain]
* sample rock (code cell 8): Navigable terrain is best identified in LAB color scheme on B channel. For the threshold I used the following bounds (150, 255):
![alt text][samplerock]

obstacles will be defined as everything else. For illustration please refer to code/perception.py lines 125 and 126.

#### Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.

Updated process_image() function located in code cell 12 of ipython book. In order to process image correctly I added the following steps:
1) Apply color threshold to identify navigable and a rock
2) Apply perspective transform and remove values hallows generated by perspective transform
3) Convert thresholded image pixel values to rover-centric coords
4) Convert rover-centric pixel values to world coords
5) Update worldmap (to be displayed on right side of screen)

In addition, I added information related to rock sample on the resulting image. This task was accomplished by identifying sample rock and calculating the distance and the angle to the rock using detect_rock() function (in code cell 10).

An example of the resulting output presented below:
![alt text][process_img]

* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

The result of processing provided images with moviepy presented below:
[link to my video result](./output/test_mapping.mp4)



### Autonomous Navigation / Mapping

#### Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook).

Updated perception_step() function is located in code/perception.py lines 108 - 152.

Main changes:
1) Detect navigable terrain - achieved through a dedicated function detect_navigable_terrain(R.img)
2) Detect rock sample - achieved through a dedicated function detect_rock(R.img, min_points_to_detect)
3) Visualize results - updated Rover.vision_image in lines 122 - 129
4) Convert rover-centric pixel values to world coordinates - lines 131 - 140
5) Update world map - lines 141 - 146

#### Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands.

In order to achieve targeted at least 40% of the environment with 60% fidelity I made the following changes:
1) Added Rover.est_steering() method - designed to provide average angle of the widest terrain area. Average steering angle provides a good starting point for picking a direction. However, this approach is not a good fit when Rover encounters 2 paths. The solution proposed in this method splits terrain into 2 parts and chooses the wides path. For more details please refer to drive_rover.py lines 80 - 98.
2) Applied Rover.est_steering() to steering angle calculation when Rover is moving forward - descision.py line 29.

In addition, I increased maximum speed and acceleration (lines 65 and 56 respectively) to speed up the data collection.

#### Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

As a result, Rover achieved approximately 60% coverage at 65% fidelity rate. A video illustration is located below:
[link to my video result](./output/autonomous.mp4)


**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

For simulation I used provided Roversim.app. I ran this simulation using 640 x 480 screen resolution and fastest graphics quality. Resulting output was in 11 - 14 FPS range.


## Other considerations and further steps

Overall this approach offers a good start. However it can be improved in the following areas:
1) Perspective transform: My current approach relies of fixed source and destination point for the description of a plane needed for perspective transform. As a result, the transformation process can be incorrect at various slopes of the navigable terrain. Addition challenge represent the acceleration and breaking point. A possible solution for this problem could be an automated method for approximating the best plane to describe the navigable terrain.
2) Differentiation between navigable terrain, sample rocks and obstacles: this process is currently done relying on week filters - color channels and thresholding. These filters may lead to incorrect representation in various whether lighting conditions. In order to overcome this problem, I would recommend to explore deep learning methods that can help to automate a process of creating filters. A good next step here would be to train a segmentation net.
3) Decision making - currently our Rover relies on a rigid set of rules. These rules produce descent result. However, I believe that these rules can be improved significantly by using machine learning. One of the solutions can be to train a decision tree algorithm based on observations. An alternative solution would be to use recurrent nets or reinforcement learning to achieve optimal result.
