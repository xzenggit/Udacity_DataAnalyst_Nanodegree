# 3) Count tags in dataset
import xml.etree.cElementTree as ET
import pprint

dataset = 'raleigh_north-carolina.osm'

def count_tags(filename):
        '''Iterative parsing to find tags and corresponding numbers.'''
        tree = ET.parse(filename)
        root = tree.getroot()
        dict_tags = {}
        return recursive_count(root, dict_tags)


def recursive_count(root, dict_tag):
    '''Recursively count the number of tags in a root.'''

    if root.tag in dict_tag:
        dict_tag[root.tag] += 1
    else:
        dict_tag[root.tag] = 1

    if len(root) != 0:
        for child in root:
            recursive_count(child, dict_tag)
    return dict_tag

tags = count_tags(dataset)
pprint.pprint(tags)

# 4) Check if there are any potential problems for each "<tag>"
'''
  "lower", for tags that contain only lowercase letters and are valid,
  "lower_colon", for otherwise valid tags with a colon in their names,
  "problemchars", for tags with problematic characters, and
  "other", for other tags that do not fall into the other three categories.
'''
import re

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

#dataset = 'raleigh_north-carolina_sample.osm'
dataset = 'raleigh_north-carolina.osm'

def key_type(element, keys):
    if element.tag == "tag":
        if lower.match(element.attrib['k']):
            keys['lower'] += 1
        elif lower_colon.match(element.attrib['k']):
            keys['lower_colon'] += 1
        elif problemchars.match(element.attrib['k']):
            keys['problemchars'] += 1
        else:
            # print element.attrib
            keys['other'] += 1

    return keys


def check_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys

keys = check_map(dataset)
pprint.pprint(keys)