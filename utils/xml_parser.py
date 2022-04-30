import xml.etree.ElementTree as ET
import os


def parse(filename):
    root = ET.parse(filename)
    
    annotation_dict = {}
    
    filename, _extension = os.path.splitext(os.path.basename(filename))
    
    annotation_dict[filename] = list()
    for child in root.findall('object'):
        obj_dict = {}
        obj_dict['name'] = child.find('name').text
        
        pose = child.find('pose')
        occulted = child.find('occulted')
        obj_dict['pose'] = pose.text if pose != 'Unspecified' else None
        obj_dict['truncated'] = child.find('truncated').text
        obj_dict['occulted'] = occulted.text if occulted != None else None
        obj_dict['difficult'] = int(child.find('difficult').text)
        
        bndbox = child.find('bndbox')
        obj_dict['xmin'] = int(bndbox.find('xmin').text)
        obj_dict['xmax'] = int(bndbox.find('xmax').text)
        obj_dict['ymin'] = int(bndbox.find('ymin').text)
        obj_dict['ymax'] = int(bndbox.find('ymax').text)
        
        annotation_dict[filename].append(obj_dict)
    
    return annotation_dict


def from_dir(path):
    annotation_dict = {}
    
    for f in os.listdir(path):
        if f.endswith('.xml'):
            annot = parse(path + f)
            filename, _extension = os.path.splitext(os.path.basename(f))
            annotation_dict.update(annot)
    return annotation_dict

if __name__ == '__main__':
    obj = from_dir('./data/annotations/')
    print(obj)      