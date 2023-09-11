## Project Name 
HOG Cat Detection <br>

## Summary
Using the Histogram of Oriented Gradients descriptor, determined whether image contains a cat
or not.

## Steps
0. Obtain data from PASCAL VOC 2012 and get positive (cat) and negative (no cat)
images. 
<br>
<p align="center">
<img src="readme_images/cats.png" width="400"/>
<img src="readme_images/non_cats.png" width="400"/>
</p>

1. Take x, y image gradients. <br>
2. Compute magnitude and orientation of gradient at each pixel location. <br>
3. Group pixels into cells (e.g. cell= 8 pixels $\times$ 8 pixels). <br>
4. For each cell, create a histogram of gradient orientation weighted by magnitude. <br>

<p align="center">
<img src="readme_images/nyuma.jpeg" width="300"/>
<img src="readme_images/nyuma_gradient.png" width="300"/>
<img src="readme_images/gradient.png" width="300"/>
</p>

5. Group the cells into overlapping blocks (e.g. 2 cells $\times$ 2 cells). <br>
6. For each block, concatenate the histograms into a single vector and normalize the obtained vector. <br>
7. Concatenate the block vectors and obtain a feature vector for the image.
8. Train and test with SVM.

<p align="center">
<img src="readme_images/svm_result.png" width="500"/>
</p>

