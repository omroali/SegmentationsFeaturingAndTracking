#import "@preview/problemst:0.1.0": pset

#show: pset.with(
  class: "Computer Vision", student: "Omar Ali - 28587497", title: "Assignment 1", date: datetime.today(),
)

= Image Segmentation and Detection 
In this task, 3 balls from some images needed to be extracted from a set of 62 images and segmented into a mask. The balls were of three types: American Football, Football, and Tennis Ball. The images represent a sequence or frames where the balls are in motion. The task was to segment the balls from the background and calculate the Dice Similarity Score (DS) of the segmented balls against the ground truth mask of the balls.

== Ball Segmentation
In this section, I will demonstrate how the balls were segmented from the background using a series of image-processing techniques. 
The steps are as follows:
#figure(
  table(
    columns: 4, stroke: none, 
    figure(image("./assets/frame-85.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/2-hsv.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/3-intensity.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/4-gblur.png", width: 90%,)),
    [Original \ Step 1 \ Grab the original image from path],[HSV \ Step 2 \ Converted the RGB image into HSV colour space], [Intensity \ Step 3 \ Extracted the Intensity Channel],[GBlur \ Step 4 \ Apply Gaussian Blur with k=[3x3] with 2 iterations],
      [],[],[],[],
    figure(image("./assets/segmentation/steps/5-blur.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/6-adap_gaus_thresh.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/7-open.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/8-dilate.png", width: 90%,)),
    [Median Blur \ Step 5 \ Apply Median Blur with k=[3x3] with 2 iterations],[Adaptive Gaussian Threshold \ Step 6 \ With Blocksize = 19 and c = 5],[Opening \ Step 7 \ to disconnecting white pixels k  = (4,4) with 5 iterations],[Dilate \ Step 8 \ expanding white pixels to remove noisy black holes holes k = (3,3) with 4 iterations ],
    [],[],[],[],
    figure(image("./assets/segmentation/steps/9-erode.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/10-fill.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/11-dilate_and_erode.png", width: 90%,)),
    figure(image("./assets/segmentation/steps/12-contour.png", width: 90%,)),
    [Erode \ Step 9 \ re-shrinking the white pixels to let the balls begin to connect k = (3,3) with 2 iterations],[Fill \ Step 10 \ Fill the enclosed black areas],[Dilate and Erode \ Step 11 \ series of dilations and erosions to remove the noisy black pixels whislt maintaining \*],[Contour \ Step 12 \ find the contours of the (bitwise_not) image],
    [],[],[],[],
  ),
)
At this stage, step 12, it can be seen that the balls have been segmented from the background. However, this is only the case in the particular sample, frame 85. In the images below, Frame 100 is re-displayed to show that the ball segmentation still had artifacts that needed to be removed. Two main culprits were the shadows of the American Football in the latter frames, as well as some sections of the background that were not yet caught by the segmentation process so far. 

In order to remove the shadow, it was found that the intensity channel of the HSV image was quite effective and computationally cheap to apply.

#figure(
  table(
    columns: 3, stroke: none, 
    figure(image("./assets/segmentation/steps/12-contour_100.png", width: 100%,)),
    figure(image("./assets/segmentation/steps/13-thresh_image_100.png", width: 100%,)),
    figure(image("./assets/segmentation/steps/13-thresh_image_bitwise_and.png", width: 100%,)),
    [Contour \ Step 12 \ Grab the original image from path],
    [Thresholing Intensity \ Step 13.1 \ A threshold of 0.4 was applied to the intensity image from step 3 to remove shadows],
    [Combine \ Step 13.2 \ The contour and the thresholded intensity image were combined],
  ),) 

The inverse of the thresholded intentity, in step 13.1, was combined with the contour using a 'bitwise and' operation. 
```python
combo = cv2.bitwise_not(cv2.bitwise_or(thresh_intensity_image, contour_image)
```

#figure(
  table(
    columns: 3, stroke: none, 
    figure(image("./assets/segmentation/steps/14-convex_hull.png", width: 100%,)),
    figure(image("./assets/segmentation/steps/16-final.png", width: 100%,)),
    [],
    [Convex Hull \ Step 14 ],
    [Circularity \ Step 15 ],
  ),)

Finally, a convex hull was applied to the contours to remove sharp edges and improve the circularity of the contours. The circularity of the contours was calculated and only contours with a circularity greater than 0.72 were kept.


== Dice Similarity Score
The Dice Similarity Score (DS) of every segmented ball image was compared against the ground truth mask of the ball image. The Dice Similarity Score (DS) is defined as:

$ "DS" = (2*|M sect S|) / (|M| + |S|) $

where M is the ground truth mask and S is the segmented mask. The DS score ranges from 0 to 1, where 1 indicates a perfect match between the two masks.

A bar chart of the DS has been plotted for every frame. The lowest recorded score is 0.82 for frame 93 and the highest is 0.96 for frame 78.

#figure(
    image("./assets/dice_score_barchart.png", width: 100%,
  )
)<seg_barchart>

Additionally, a violin plot, was used to display the distribution of the DS score across all the frames, it can be seen that there is quite a skew of the data towards the higher range of the DS score, with the greatest proportion of the data sitting around the 0.93-0.94 range. 

#figure(
    image("./assets/dice_score_violin.png", width: 100%,
  ), 
)<seg_boxplot>

Below are the best and worst frames in terms of the Dice Similarity Score achieved by the segmentation process.
#figure(
  table(
    columns: 3, stroke: none, 
    [],[Best Frame 78 - DS 0.961],[],
      [Segmented Image], [Ground Truth], [Difference],
    figure(image("./assets/segmentation/best/frame-78.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-78_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-78_ds_0.9608055770720372.jpg", width: 100%,)),
    [],[],[],
    [],[Worst Frame 78 - DS 0.961],[],
    figure(image("./assets/segmentation/worst/frame-93.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-93_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-93_ds_0.8195569136745607.jpg", width: 100%,)),
    [Segmented Image 93], [Ground Truth 93], [DS 0.820],
  ),
)

== Discussion
The implementation was found to be quite good, where the DS scores had a maximum of 0.96, minimum of 0.82, mean of 0.91 and standard deviation of 0.04. However, the implementation is quite clunky and very tuned for this task. 

Overall, the number of steps in order to reach the solution is quite large, at 15 steps. The additional processing was ultimately due to a whack-a-mole situation, where refinements in one area of the task cause another area to worsen. This makes this solution temperamental and not very robust to changes in the input data, however, for very refined results, this is likely to be a similar case for most image segmentation methodologies that do not more complex models such as deep learning.

In the initial processing, it was found that the Intensity value provided a very good initial starting point, as opposed to using grayscale, as the intensity of the balls has a very distinct value compared to the background. The Gaussian Blur and Median Blur were  very effective at removing the nose from this image, whilst maintaining the edges of the balls. The Adaptive Gaussian Threshold (AGT) was effective in detecting the edges in the image, compared to a standard threshold, because it takes into consideration a normalised local area for its thresholding calculation. 
[[TODO? possibly discuss the C and blocksize values]]

The range of morphological filters was essential in getting a cleaner mask of the balls, with just the right amount of erosion and dilation to remove the noise from the background. Different ranges of kernel sizes were used for the erosion and dilation, however all used some scale of the `MORPH_ELLIPSE` structuring element. It is important to note that in the first applications of the morphological filters, the region being eroded or dilated was the background, not the balls. This was because the image had not yet been inverted and so the balls were considered the background.

Once the morphological filtering was complete, the balls were segmented using the `opencv.findContours` function. As mentioned, this was found to not be refined enough, so the intensity channel was used to remove the shadows from the American Football, a convex hull was applied to the contours to connect sharp edges, and the circularity of the contours where only contours with a circularity greater than 0.72 were kept.

There are certainly some improvements that could be made, and would have possibly decided to take a different approach had this task been reattempted. A possible alternative first step could be to make use of the colour channels of the image to segment the balls. This could be done by applying a threshold to the colour channels and then combining the results. This would have been a more robust approach where the main challenge would be the distinction of the white football from the walls of the room. Additionally, considering the constrained environment, it could have been possible to create a mask of the problematic parts of the room and have these be removed from the image before segmentation.

= Feature Calculation (30%)

== Shape Features
In this section, I will demonstrate how a range of shape features (Solidity, Non-compactness, Circularity, Eccentricity) can be calculated from the contours of the segmented balls.

In order to get the contours of the segmented balls, I made use of the openCV 'findContours' function which returns a list of contours and a hierarchy for all the contours in the image. 

```python
img = cv2.imread(image)
img = cv2.bitwise_and(cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR), img)
contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
```

Each contour is split into the corresponding balls in order to combine the contours of the same ball into an array, this is done by exploiting the areas of each ball. The details of the code can be explored as needed, however the core logic is as follows:

```python
area = cv2.contourArea(contour)
if area > 1300:  # football
    append_ball = BALL_LARGE
elif area > 500:  # soccer_ball
    append_ball = BALL_MEDIUM
else:  # tennis ball
    append_ball = BALL_SMALL
```

Using this, the features can be evaluated for every ball.

=== Circularity
Circularity is defined as the ratio of the area of the circle with the same perimeter as the object to the area of the object. For a given contour C, the circularity is calculated by:
```python
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, closed=True)
circularity = (4 * math.pi * area) / (perimeter**2)
```
#figure(
  table(
    columns: 3, stroke: none, 
    figure(image("./assets/features/Circularity_histogram_American_Football.png", width: 100%,)),
    figure(image("./assets/features/Circularity_histogram_Football.png", width: 100%,)),
    figure(image("./assets/features/Circularity_histogram_Tennis.png", width: 100%,)),
  ),
)

Both the Football and the Tennis ball have a higher circularity than the American Football. This is expected as the American Football has a more elongated shape compared to the other two balls. Visually, the football has a smaller variance in circularity compared to the tennis ball. This is likely due to the relative size of the football compared to the tennis ball, where the football is larger and has a more consistent shape from the perspective of an image and will not suffer from distortion and be impacted by smaller pixel ranges.

=== Eccentricity 
Eccentricity is the ratio of the distance between the foci of the ellipse to the major axis length. For a given contour C, the eccentricity is calculated by:
```python
ellipse = cv2.fitEllipse(contour)
a = max(ellipse[1])
b = min(ellipse[1])
eccentricity = (1 - (b**2) / (a**2)) ** 0.5
```
#figure(
  table(
    columns: 3, stroke: none, 
    figure(image("./assets/features/Eccentricity_histogram_American_Football.png", width: 100%,)),
    figure(image("./assets/features/Eccentricity_histogram_Football.png", width: 100%,)),
    figure(image("./assets/features/Eccentricity_histogram_Tennis.png", width: 100%,)),
  ),
)

The American Football has the highest eccentricity of all the balls, which is expected as it has a more elongated shape compared to the other two balls. The football and the tennis ball both have very distributed eccentricity values.
=== Solidity
Solidity is the ratio of the area of the object to the area of the convex hull of the object. For a given contour C, the solidity is calculated by: 
```python
area = cv2.contourArea(contour)
convex_hull = cv2.convexHull(contour)
convex_area = cv2.contourArea(convex_hull)
solidity = area / convex_area
```
#figure(
  table(
    columns: 3, stroke: none, 
    figure(image("./assets/features/Soliditiy_histogram_American_Football.png", width: 100%,)),
    figure(image("./assets/features/Soliditiy_histogram_Football.png", width: 100%,)),
    figure(image("./assets/features/Soliditiy_histogram_Tennis.png", width: 100%,)),
  ),
)

The solidity of the American Football is much higher and more consistent than the other two balls. This is likely due to the ball being larger in size and so a convex hull around the ball is likely to be more similar to the ball itself. This follows through as we see the football having a higher solidity than the tennis ball, and the tennis ball's solidity is much more distributed than the others.
=== Non-compactness
Non-compactness is the ratio of the area of the object to the area of the circle with the same perimeter as the object. For a given contour C, the non-compactness is calculated by:
```python
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, closed=True)
non_compactness = 1 - (4 * math.pi * area) / (perimeter**2)
```
#figure(
  table(
    columns: 3, stroke: none, 
    figure(image("./assets/features/Non-Compactness_histogram_American_Football.png", width: 100%,)),
    figure(image("./assets/features/Non-Compactness_histogram_Football.png", width: 100%,)),
    figure(image("./assets/features/Non-Compactness_histogram_Tennis.png", width: 100%,)),
  ),
)

The Football has the most consistent non-compactness values, [[IDK add something about non compactness]]




== Texture Features

In this section, texture features are calculated from the segmented balls. The texture features are evaluated by calculating the normalised Grey-Level Co-occurrence Matrix (GLCM) in four orientations (0°, 45°, 90°, 135°) for every individual ball. The GLCM is a matrix that describes how often different combinations of pixel intensity values occur in an image. The GLCM is calculated for each of the colour channels (red, green, blue) and for each of the four orientations then averaged across the orientations to determine the texture features of the ball. These features include Angular Second Moment, Contrast, Correlation, and Entropy.

For each feature, one colour channel is selected to demonstrate the feature calculation. The blue channel was selected for the Angular Second Moment, the red channel for Contrast, and the green channel for Correlation.

To get the GLCM, the balls were segmented in a similar way to what was described previously in the shape features section. However, this time, the mask generated was overlayed onto the original image to get the pixel values of the balls. 

#figure(
  table(
    columns: 3, stroke: none, 
  figure(image("./assets/colour_american_football_mask.png")),
  figure(image("./assets/colour_football_mask.png")),
  figure(image("./assets/colour_tennis_mask.png")),
  ))

The GLCM was calculated using the 'greycomatrix' function from the skimage library. The first row and column were both stripped away from the GLCM to remove the background noise. The GLCM was then normalised using the 'greycoprops' and the texture features were evaluated for all four orientations. These were then averaged to get the final average texture value for the ball.

=== Applied Texture Features
Below are a violin plot of the texture featuures for the three balls. The Angular Second Moment was calculated for the blue channel, the Contrast for the red channel, and the Correlation for the green channel, where the yellow plot represents the Tennis Ball, the white plot represents the Football, and the orange plot represents the American Football. 
#figure(
  table(
    columns: 5, stroke: none, 
  figure(image("./assets/features/asm_data_blue_channel.png",)),
  figure(image("./assets/features/asm_range_data_blue_channel.png")),
  figure(image("./assets/features/contrast_data_red_channel.png")),
  figure(image("./assets/features/correlation_green_channel.png")),
  ),)
In the ASM of the Blue Channel, the American Football has the highest distribution of values, followed by the Football and then the Tennis Ball. This makes sense as the American Football's orientation changes throughout the frames, and so when the ball appears more spherical in the image, the Angular Second Moment will be higher as the pixel values are more similar. The Tennis Ball has the lowest distribution and the highest ASM values, this is both due to the Tennis Ball being the smallest ball so the pixel values are more similar, and also due to the Tennis Ball having the least amount of noise in the image as it is very bright and yellow throughout the frames. The range of the ASM values shifts the tennis ball down the scale of the ASM values. [[what is the importance of this]]

// ---

// The Angular Second Moment (ASM), also known as Energy, is a measure of textural uniformity within an image. It is calculated from a Grey-Level Co-occurrence Matrix (GLCM). The ASM will be high when the image has constant or repetitive patterns, meaning the pixel intensities are similar or identical throughout the image. In contrast, the ASM will be low when the image has a lot of variation or randomness in pixel intensities.

// Yellow Tennis Ball: High ASM mean, small standard deviation (std)

// The yellow tennis ball has a high ASM mean in the blue channel, which suggests that the pixel intensities in the blue channel for the yellow tennis ball are very similar or identical throughout the image. This could be due to the yellow color having a low blue component in the RGB color model.
// The small standard deviation indicates that the pixel intensities in the blue channel for the yellow tennis ball are consistent, suggesting that the ball's color and lighting conditions are uniform.
// White Football: Low mean, low distribution (small standard deviation)

// The white football has a low ASM mean in the blue channel, which suggests that there is a lot of variation or randomness in pixel intensities in the blue channel for the white football. This could be due to the white color having a high blue component in the RGB color model, or due to variations in lighting conditions or reflections on the ball.
// The low standard deviation indicates that the pixel intensities in the blue channel for the white football are consistent, suggesting that the ball's color and lighting conditions are uniform.
// Orange American Football: Low mean, high distribution (large standard deviation)

// The orange American football has a low ASM mean in the blue channel, which suggests that there is a lot of variation or randomness in pixel intensities in the blue channel for the orange American football. This could be due to the orange color having a low blue component in the RGB color model, or due to variations in lighting conditions or reflections on the ball.
// The large standard deviation indicates that the pixel intensities in the blue channel for the orange American football vary widely. This could be due to factors such as lighting conditions, shadows, or variations in the ball's color.
// In summary, these distributions reflect the different colors of the balls and how they interact with the blue channel, as well as the consistency of the lighting conditions and color across each ball.

// ---


// The range of the Angular Second Moment (ASM) in an image indicates the spread of textural uniformity across the image.

// In other words, it shows the difference between the highest and lowest ASM values in the image. A high range would indicate that there are areas of the image with very high textural uniformity (constant or repetitive patterns) and areas with very low textural uniformity (a lot of variation or randomness in pixel intensities).

// Conversely, a low range would suggest that the textural uniformity across the image is relatively consistent, with less variation in the pixel intensities.

// In the context of the analysis you're doing, the range of the ASM values can provide insights into the variability of the texture across different parts of the balls or across different frames. For example, a high range in the ASM values for the American football might suggest that there are significant changes in its appearance (due to changes in orientation, lighting, etc.) across different frames.
// The range of the Angular Second Moment (ASM) in an image indicates the spread of textural uniformity across the image. Here's what the distributions of the RANGE ASM on the blue channel of the image for the different balls might indicate:

// Yellow Tennis Ball: Low ASM range mean, small standard deviation (std)

// The yellow tennis ball has a low ASM range mean in the blue channel, which suggests that the spread of textural uniformity across the image is low. This means that the pixel intensities in the blue channel for the yellow tennis ball are very similar or identical throughout the image. This could be due to the yellow color having a low blue component in the RGB color model.
// The small standard deviation indicates that the spread of textural uniformity across the image for the yellow tennis ball is consistent, suggesting that the ball's color and lighting conditions are uniform.
// White Football: Low ASM range mean, low distribution (small standard deviation)

// The white football has a low ASM range mean in the blue channel, which suggests that the spread of textural uniformity across the image is low. This could be due to the white color having a high blue component in the RGB color model, or due to variations in lighting conditions or reflections on the ball.
// The low standard deviation indicates that the spread of textural uniformity across the image for the white football is consistent, suggesting that the ball's color and lighting conditions are uniform.
// Orange American Football: Low (but higher than other 2) ASM range mean, high distribution (large standard deviation)

// The orange American football has a low (but higher than the other two balls) ASM range mean in the blue channel, which suggests that the spread of textural uniformity across the image is slightly higher than for the other two balls. This could be due to the orange color having a low blue component in the RGB color model, or due to variations in lighting conditions or reflections on the ball.
// The large standard deviation indicates that the spread of textural uniformity across the image for the orange American football varies widely. This could be due to factors such as lighting conditions, shadows, or variations in the ball's color.
// In summary, these distributions reflect the different colors of the balls and how they interact with the blue channel, as well as the consistency of the lighting conditions and color across each ball.

// ---

// The contrast of an image, specifically in a color channel, refers to the difference in color that makes an object distinguishable. In the red channel of an image, objects with a high amount of red will have a high intensity.

// Yellow Tennis Ball: High contrast mean, large standard deviation (std)

// The yellow tennis ball has a high contrast mean in the red channel because yellow is a combination of red and green in the RGB color model. This means that the yellow tennis ball will have a high intensity in the red channel, leading to a high contrast mean.
// The large standard deviation indicates that the pixel intensities in the red channel for the yellow tennis ball vary widely. This could be due to factors such as lighting conditions, shadows, or variations in the ball's color.
// White Football: Same mean as the yellow tennis ball, less distributed (smaller standard deviation)

// The white football having the same mean contrast as the yellow tennis ball suggests that it also has a high intensity in the red channel. This could be due to the lighting conditions or reflections on the ball.
// The smaller standard deviation indicates that the pixel intensities in the red channel for the white football are more consistent than those of the yellow tennis ball. This could suggest that the football's color and lighting conditions are more uniform.
// Orange American Football: Low mean, low distribution (small standard deviation)

// The orange American football has a low contrast mean in the red channel, which suggests that it has a lower intensity in the red channel compared to the yellow tennis ball and white football. This could be due to the orange color having less red component compared to yellow or due to different lighting conditions.
// The low standard deviation indicates that the pixel intensities in the red channel for the orange American football are consistent, suggesting that the ball's color and lighting conditions are uniform.
// In summary, these distributions reflect the different colors of the balls and how they interact with the red channel, as well as the consistency of the lighting conditions and color across each ball.

// ---

// The correlation of an image, specifically in a color channel, refers to the degree to which neighboring pixel values are related. In the green channel of an image, objects with a high amount of green will have a high intensity.

// 1. Yellow Tennis Ball: Low correlation mean, large standard deviation (std)
//    - The yellow tennis ball has a low correlation mean in the green channel because yellow is a combination of red and green in the RGB color model. This means that the yellow tennis ball will have a high intensity in the green channel, but the correlation is low. This could be due to variations in the green intensities across the ball, perhaps due to shadows, lighting conditions, or variations in the ball's color.
//    - The large standard deviation indicates that the pixel correlations in the green channel for the yellow tennis ball vary widely. This could be due to factors such as lighting conditions, shadows, or variations in the ball's color.

// 2. White Football: Middle mean, less distributed (smaller standard deviation)
//    - The white football having a middle mean correlation in the green channel suggests that it has a moderate relationship between neighboring pixel values in the green channel. This could be due to the lighting conditions or reflections on the ball.
//    - The smaller standard deviation indicates that the pixel correlations in the green channel for the white football are more consistent than those of the yellow tennis ball. This could suggest that the football's color and lighting conditions are more uniform.

// 3. Orange American Football: High mean, low distribution (small standard deviation)
//    - The orange American football has a high correlation mean in the green channel, which suggests that it has a strong relationship between neighboring pixel values in the green channel. This could be due to the orange color having less green component compared to yellow or due to different lighting conditions.
//    - The low standard deviation indicates that the pixel correlations in the green channel for the orange American football are consistent, suggesting that the ball's color and lighting conditions are uniform.

// In summary, these distributions reflect the different colors of the balls and how they interact with the green channel, as well as the consistency of the lighting conditions and color across each ball.


== Part C: Discriminative Information
The American Football has the most distincive 

Based on your visualisations in Part a) and b), discuss which features appear to
be best at differentiating between different ball types. For each ball type, are
shape or texture features more informative? Which ball type is the
easiest/hardest to distinguish, based on the calculated features? Which other
features or types of features would you suggest for the task of differentiating
between the different ball types and why?

#pagebreak()
= Object Tracking 
Implement a Kalman filter from scratch (not using any method/class from
pre-built libraries) that accepts as input the noisy coordinates [na,nb] and
produces as output the estimated coordinates [x\*,y\*]

SEE Workshop 4 for the Kalman Filter implementation in Matlab

Constant Velocity motion model F Constant time intervals Δt = 0.5 Cartesian
observation model H

covariance matrices Q and R in task sheet

== Part 1
// Plot the estimated trajectory of coordinates [x*,y*], together with the real
[x,y] and the noisy ones [a,b] for comparison.

Discuss solution

== Part 2
Assess the quality of the tracking by calculating the mean and standard
deviation of the Root Mean Squared error (include the mathematical formulas you
used for the error calculation in your report)

Compare both noisy and estimated coordinates to the ground truth. Adjust the
parameters associated with the Kalman filter, justify any choices of
parameter(s) associated with Kalman Filter that can give you better estimation
of the coordinates that are closer to the ground truth.

Discuss and justify your findings in the report.


#pagebreak()
= Appendix.

#figure(
  table(
    columns: 3, stroke: none, 
    figure(image("./assets/segmentation/worst/frame-93.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-93_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-93_ds_0.8195569136745607.jpg", width: 100%,)),
    [Segmentation-93], [GT-93], [Difference 0.820],[],[],[],
    figure(image("./assets/segmentation/worst/frame-115.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-115_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-115_ds_0.822869077143976.jpg", width: 100%,)),
    [Segmentation-115], [GT-115], [Difference 0.823],[],[],[],
    figure(image("./assets/segmentation/worst/frame-94.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-94_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-94_ds_0.8287037037037037.jpg", width: 100%,)),
    [Segmentation-94], [GT-94], [Difference 0.829],[],[],[],
    figure(image("./assets/segmentation/worst/frame-116.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-116_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-116_ds_0.8313458262350937.jpg", width: 100%,)),
    [Segmentation-116], [GT-116], [Difference 0.831],[],[],[],
    figure(image("./assets/segmentation/worst/frame-103.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-103_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/worst/frame-103_ds_0.8532547699214366.jpg", width: 100%,)),
    [Segmentation-103], [GT-103], [Difference 0.853],[],[],[],
  ),
  caption: [Worst Frames],
)
#figure(
  table(
    columns: 3, stroke: none, 
    figure(image("./assets/segmentation/best/frame-78.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-78_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-78_ds_0.9608055770720372.jpg", width: 100%,)),
    [Segmentation-78], [GT-78], [Difference 0.961],[],[],[],
    figure(image("./assets/segmentation/best/frame-62.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-62_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-62_ds_0.9565345513481656.jpg", width: 100%,)),
    [Segmentation-62], [GT-62], [Difference 0.957],[],[],[],
    figure(image("./assets/segmentation/best/frame-70.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-70_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-70_ds_0.9555305827580671.jpg", width: 100%,)),
    [Segmentation-70], [GT-70], [Difference 0.956],[],[],[],
    figure(image("./assets/segmentation/best/frame-69.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-69_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-69_ds_0.9537412757669209.jpg", width: 100%,)),
    [Segmentation-69], [GT-69], [Difference 0.954],[],[],[],
    figure(image("./assets/segmentation/best/frame-61.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-61_GT.png", width: 100%,)),
    figure(image("./assets/segmentation/best/frame-61_ds_0.9512928442573662.jpg", width: 100%,)),
    [Segmentation-61], [GT-61], [Difference 0.951],[],[],[],

  ),
  caption: [Best Frames],
  )


== All Features
#figure(image("./assets/features/asm_data.png", width: 80%,))
#figure(image("./assets/features/asm_range_data.png", width: 80%,)),
#figure(image("./assets/features/contrast_data.png", width: 80%,)),
#figure(image("./assets/features/correlation_data.png", width: 80%,)),




