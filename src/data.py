import xml.etree.ElementTree as ET
import numpy as np

def contains_category(xml_file_path, category='cat'):
    """
    Check if the image contains desired category.

    Args:
        xml_file_path: path to xml file containing annotations
        category: desired category
    
    Returns:
        Returns True if image contains the desired category
    """

    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name == category:
            return True
        
    return False

def get_bounding_boxes(xml_file_path, category='cat'):
    """
    Go through the PASCAL VOC dataset and check which images contain the desired 
    category. Then　obtain its/their bounding boxes.

    Args:
        xml_file_path: xml file in PASCAL VOC format
    
    Returns:
        list of cat bounding boxes
    """

    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    bounding_boxes = []

    for obj in root.iter('object'):
        name = obj.find('name').text
        if name == category:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bounding_boxes.append((xmin, ymin, xmax, ymax))

    return bounding_boxes

def get_positive_region(image, bounding_boxes):
    """
    Given an image and its positive bounding boxes, return list of positive parts
    of the image.

    Args: 
        image: grayscale image
        bounding_boxes: list of bounding boxes (xmin, ymin, xmax, ymax)
    
    Returns:
        positive_images: list of positive parts of image
    """

    positive_regions = []
    for bounding_box in bounding_boxes:
        xmin, ymin, xmax, ymax = bounding_box
        positive_regions.append(image[ymin:ymax, xmin:xmax])

    return positive_regions

def get_random_region(image, min_width, min_height):
    """
    Given an image returns a random cropped region.

    * If the original image has height or width smalled thanm the desired minimum,
    simply return the original image.

    Args:
        image: grayscale image
        min_width: min width for random crop
        min_height: min height for random crop

    Returns:
        Random crop of image (if possible) or original image
    """
    M, N = image.shape

    if M < min_height or N < min_width:
        return image
    
    x_min = np.random.randint(0, N - min_width)
    x_max = np.random.randint(x_min+min_width, N)

    y_min = np.random.randint(0, M - min_height)
    y_max = np.random.randint(y_min+min_height, M)

    return image[y_min:y_max, x_min:x_max]

    


