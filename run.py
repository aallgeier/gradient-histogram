# python -m pip install <package>
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import os
from sklearn import svm
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.hog_descriptor import create_image_histograms
from src.hog_descriptor import create_descriptor

from src.data import get_bounding_boxes
from src.data import get_positive_region
from src.data import contains_category
from src.data import get_random_region

from src.utils import plot_grad_histogram_grid


ANNOT_ROOT = "../Data/VOCdevkit/VOC2012/Annotations"
IMAGE_ROOT = "../Data/VOCdevkit/VOC2012/JPEGImages"

# Get annotations containing cat
annot_list_temp = list(sorted(Path(ANNOT_ROOT).glob("./*.xml")))
cat_annot = []
non_cat_annot = []
for annot_p in annot_list_temp:
    if contains_category(annot_p, category="cat"):
        cat_annot.append(annot_p)
    else:
        non_cat_annot.append(annot_p)

# Get positive regions and convert to hog descriptor.
cell_size = 8
block_size = 2

descriptor_labels = []

print("...preparing positive descriptors ")

positive_descriptors = []
for annot_p in tqdm(cat_annot):
    # Get grayscale image
    image = cv2.imread(os.path.join(IMAGE_ROOT, annot_p.name[:-4] + ".jpg"))[:, :, ::-1].astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get regions contain cats
    bounding_boxes = get_bounding_boxes(annot_p, category='cat')
    positive_regions = get_positive_region(gray, bounding_boxes)

    # Create HOG descriptor
    for im in positive_regions:
        image = cv2.resize(im, (256, 256))
        histogram_grid, bins = create_image_histograms(image, cell_size=cell_size)
        blocks = create_descriptor(histogram_grid, block_size=block_size, step_size=1)
        
        descriptor_labels.append(1)
        positive_descriptors.append(blocks.flatten())

print("...positive descriptors ready", len(positive_descriptors))

print("...preparing negative descriptors ")
negative_descriotors = []
for i in tqdm(range(len(positive_descriptors))):
    # Get negative image
    annot_p = np.random.choice(non_cat_annot)
    image = cv2.imread(os.path.join(IMAGE_ROOT, annot_p.name[:-4] + ".jpg"))[:, :, ::-1].astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # choose random region from image
    random_region = get_random_region(gray, 100, 100)

    # Get descriptor
    image = cv2.resize(random_region, (256, 256))
    histogram_grid, bins = create_image_histograms(image, cell_size=cell_size)
    blocks = create_descriptor(histogram_grid, block_size=block_size, step_size=1)

    descriptor_labels.append(0)
    negative_descriotors.append(blocks.flatten())

print("...negative_descriotors ready", len(negative_descriotors))

data_X = np.vstack(positive_descriptors+negative_descriotors)
data_y = np.array(descriptor_labels)

# Shuffle data for split
inds = np.arange(len(data_X))
np.random.shuffle(inds)

X_train = data_X[inds[:len(inds)//2]]
y_train = data_y[inds[:len(inds)//2]]

X_test = data_X[inds[len(inds)//2:]]
y_test = data_y[inds[len(inds)//2:]]

clf = svm.SVC()
print("...fitting SVM")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))











