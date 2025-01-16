# Object-Segmentation-And-Classification
This repository features a project on image segmentation, feature extraction, and classification of objects such as caps, watches, mice, and controllers, using computer vision techniques.

The dataset consists of images with a white background. 
![Classes2](https://github.com/user-attachments/assets/035ddc1b-76bb-4e72-9fe4-42348e72c02d)

To process the dataset, I used semantic segmentation, classifying each pixel as either part of an object or the background. For postprocessing, I extracted the largest contours (objects) and applied the Opening operator, which involves dilation followed by erosion. Finally, I filled the holes in the segmented objects. Some results are shown below.

![Seg_ex_gorra](https://github.com/user-attachments/assets/8065b6c5-faac-4390-af2c-6fb5a0e515df)
![Seg_ex_mouse](https://github.com/user-attachments/assets/fb64a5c0-ed38-4661-9786-dd9e6bead1df)

I extracted 11 features, including 4 shape features and 7 Hu invariant moments. For classification, I used a multi-layer perceptron with three layers containing 10, 256, and 128 nodes, respectively. The mean accuracy of the 5-fold cross-validation was 1. Below is the resulting confusion matrix:

![CF2](https://github.com/user-attachments/assets/ec941eec-a611-4efc-81c9-37aeb8a56921)
