# Iris segmentation
## Abstract
This project aims to accurately segment the iris and pupils in an image.

## Limitations and assumptions
 There are a few limitations that limit my ability to gain optimal results.
### Limitations
 - As far as I know there is no labeled database for iris segmentation in selfie settings
 - No easily available dataset for eye detection or iris segmentation, These databases do exist, but they are not publicly available, and using it requires the consent of the database owner
- I haven't found a deep learning model for eye segmentation.
### Assumptions:
 - The iris is clearly visible in the image
 - There is only one subject in the image

# Project proposal

The project is divided into two main parts, the detection phase, and the segmentation phase.

## Detection
The detection phase goal is to give the bounding box that contains each eye in the image.
there are two detectors that I've used in this project:
### OpenCV haar cascade
Object detection using Haar feature-based cascade classifiers proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001.
 
This detector is relatively weak and there are some post-processing that was used to increase its accuracy  

### Dlib face keypoint detector
This detector is the DLIB face landmark model, this model is much more accurate and users are encouraged to use it.


## Segmentation
As I could not find any Deep Learning architecture to solve this problem I looked for a more traditional approach. 
I divided this mission into two parts:
- iris localization
- iris segmentation
### **Iris localization**
I've implemented the paper :
[A Robust Scheme for Iris Segmentation in MobileEnvironment](/A_Robust_Scheme_for_Iris%20_Segmentation_in_MobileEnvironment.pdf)
which combines the Daguman algorithm for iris localization and adds more steps for better results in mobile camera settings.
the result of this algorithm is a circle location and radius containing the iris in the image.
### **Iris segmentation**
As the iris is not always completely round, I’ve used
the GrabCut algorithm to fit the mask in the image better.

## Ground Truth:
In order to evaluate the results I've modified the OpenCV's GrabCut annotation tool to create ground truth results for each subject.
[annotations](/gt/annotations) 
[grab cut tool](utils/grabcut.py)

## Evaluation  
Using the ground truth I can evaluate the result of each run.
the metrics that I'm interested in are:
- Precision
- Recall: Percentage of Iris detected
- Intersection over union: The combination between the above two metrics.

## Results:

[opencv](out/opencv_haar/debug/evaluation/results.csv) 

[dlib](out/dlib/debug/evaluation/results.csv) 

The visual results can be seen in:
[results](out) 
The directories are organized as follows:
- run name  
    - debug
        - evaluation: The evaluation metrics of the run
        - images: Shows all the phases of the algorithm for each subject in one image.
    - blend: shows the final result combining the mask and the image
    - mask the final mask for each subject.











