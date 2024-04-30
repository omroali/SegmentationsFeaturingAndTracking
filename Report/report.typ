#import "@preview/problemst:0.1.0": pset

#show: pset.with(
  class: "Computer Vision",
  student: "Omar Ali - 28587497",
  title: "Assignment 1",
  date: datetime.today(),
)

#let deriv(num, dnm)=[$ (d num) / (d dnm) $]
$ deriv(f(x), x)&= lim_(Delta x arrow 0) (f(x + Delta x) - f(x)) / (Delta x). $

= Image Segmentation and Detection (40%)

== Part A 
Automated ball objects segmentation. For each image, automatically segment the balls
from background. 



== Part B
Segmentation evaluation. For each ball image, calculate the Dice Similarity Score (DS)
which is defined in Equation 1; where M is the segmented ball region you obtained from Task
1, and S is the corresponding ground-truth binary ball mask.

$ "DS" = (2*|M sect S|) / (|M| + |S|) $



#line(length: 100%)

Your report should include: \
1) for all the 63 ball images, please provide a bar graph with x-axis\
representing the number of the image, and y-axis representing the corresponding DS. \ 2) calculate the mean and standard deviation of the DS for all the 63 images
\ 3) briefly
describe and justify the implementation steps. Please note that you are required to show the
best 5 and worst 5 segmented ball images (along with the corresponding ball GT mask
images) in the Appendix.

#line(length: 100%)


#pagebreak()
= Feature Calculation (30%)

== Part A
=== shape features 
For each of the ball patches, calculate four different shape features discussed in the lectures 

=== solidity, non-compactness, circularity, eccentricity
Plot the distribution of all four features, per ball type.

== Part B: Texture Features
Calculate the normalised grey-level co-occurrence matrix in four orientations (0°, 45°, 90°, 135°) for the patches from the three balls, separately for each of the colour channels (red, green, blue). 

For each orientation, calculate the first three features proposed by Haralick et al. 
  1. Angular Second Moment
  2. Contrast
  3. Correlation 
and produce perpatch features by calculating the feature average and range across the 4 orientations. Select one feature from each of the colour channels and plot the distribution per ball type.

== Part C: Discriminative Information
Based on your visualisations in Part a) and b), discuss which features appear to be best at differentiating between different ball types. For each ball type, are shape or texture features more informative? Which ball type is the easiest/hardest to distinguish, based on the calculated features? Which other features or types of features would you suggest for the task of differentiating between the different ball types and why?




#pagebreak()
= Object Tracking (30%)
Implement a Kalman filter from scratch (not using any method/class from pre-built libraries) that accepts as input the noisy coordinates [na,nb] and produces as output the estimated coordinates [x\*,y\*]

Constant Velocity motion model F
Constant time intervals Δt = 0.5
Cartesian observation model H

covariance matrices Q and R in task sheet

== Part 1 
Plot the estimated trajectory of coordinates [x*,y*], together with the real [x,y] and the noisy ones [a,b] for comparison. 

Discuss solution

== Part 2
Assess the quality of the tracking by calculating the mean and standard deviation of the Root Mean Squared error (include the mathematical formulas you used for the error calculation in your report)

Compare both noisy and estimated coordinates to the ground truth. Adjust the parameters associated with the Kalman filter, justify any choices of parameter(s) associated with Kalman Filter that can give you better estimation of the coordinates that are closer to the ground truth. 

Discuss and justify your findings in the report.