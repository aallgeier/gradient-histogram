import xml.etree.ElementTree as ET

def get_bounding_box(xml_path):
    """
    Given a xml file in PASCAL VOC format, provide coordinates of the bouding box.

    Args:
        xml_path: path to xml file containing annotations

    Returns:
        xmin, xmax, ymin, ymax: min/max coordinates of the bounding box
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for object in root.iter('object'):
        obj_name = object.find('name').text
        if obj_name == 'person':
            xmin = int(object.find('bndbox/xmin').text)
            ymin = int(object.find('bndbox/ymin').text)
            xmax = int(object.find('bndbox/xmax').text)
            ymax = int(object.find('bndbox/ymax').text)
    
    return xmin, xmax, ymin, ymax