# 2) Wrangle the data and transform the shape of the data model
import re
import json
import pprint
import xml.etree.cElementTree as ET

#dataset = 'raleigh_north-carolina_sample.osm'
dataset = 'raleigh_north-carolina.osm'


# For different types of tags
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

# For address and other tag types
re_addr = re.compile(r'^addr:[^:]*$')
re_xxx = re.compile(r'^[^:]*:[^:]*$')

# Fix unexpected street type
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place",
            "Square", "Lane", "Road","Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = { "St": "Street",
            "St.": "Street",
            "Rd": "Road",
           "Rd.": "Road",
           "Ct": "Court",
           "Ct.": "Court"
            }


def update_name(name, mapping):
    '''Update name style in mapping dictionary.'''
    tmp = name.split()
    if tmp[-1] in mapping:
        tmp[-1] = mapping[tmp[-1]]
    name = " ".join(tmp)

    return name

def shape_element(element):
    '''Shape the element in XML into data model format.'''
    node = {}
    if element.tag == "node" or element.tag == "way" :
        node['id'] = element.attrib['id']
        node['type'] = element.tag
        if 'visible' in element.attrib:
            node['visible'] = element.attrib['visible']
        # For items in CREATED
        node['created'] = {}
        for x in CREATED:
            if x in element.attrib:
                node['created'][x] = element.attrib[x]
        if 'lon' in element.attrib:
            node['pos'] = [float(element.attrib['lat']), float(element.attrib['lon'])]

        for tag in element.iter('tag'):
        # If not in problemchars
        #if 'k' in element.attrib:
          if not problemchars.match(tag.attrib['k']):
            # If in addr:xxx form

            if re_addr.match(tag.attrib['k']):

                if 'address' not in node:
                    node['address'] = {}
                # print tag.attrib['k']
                if tag.attrib['k'][5:] == 'street':
                    node['address'][tag.attrib['k'][5:]] = update_name(tag.attrib['v'],
                                                                      mapping)
                elif tag.attrib['k'][5:] == 'postcode':
                    node['address'][tag.attrib['k'][5:]] = tag.attrib['v'][0:5]
                else:
                    node['address'][tag.attrib['k'][5:]] = tag.attrib['v']

            # If not in xxx:xxx form
            elif re_xxx.match(tag.attrib['k']):
                tmp = tag.attrib['k'].index(':')
                if tag.attrib['k'][:tmp] not in node:
                    node[tag.attrib['k'][:tmp]] = {}
                node[tag.attrib['k'][:tmp]][tag.attrib['k'][tmp+1:]] = tag.attrib['v']

        tmp = []
        for tag in element.iter('nd'):
            if 'ref' in tag.attrib:
                tmp.append(tag.attrib['ref'])
        if len(tmp) > 0:
            node['node_refs'] = tmp

        return node
    else:
        return None


def process_map(file_in, pretty = False):
    # You do not need to change this file
    file_out = "{0}.json".format(file_in)
    data = []
    with open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data


if __name__ == "__main__":
    data = process_map(dataset)
    pprint.pprint(data[0:10])

    # Print sample data
    #for x in data:
    #    if 'address' in x:
    #        pprint.pprint(x['address'])