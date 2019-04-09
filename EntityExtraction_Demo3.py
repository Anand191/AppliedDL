import os
from pprint import pprint
import json

from file_handling import FileHandler
from rosette_call import extract_entities

path = './resources/EC_regulations/CELEX_32013R1375_EN_TXT.html'
json_path = './resources/EC_regulations/out/CELEX_32013R1375_EN_TXT.json'
fh = FileHandler()
ee = extract_entities()

html_file = open(path)
parsed = fh.html2text(html_file)
paragraphs = parsed.split('\n')

para_level_entities = 0
for i, sent in enumerate(paragraphs):
    print('para {} is:'.format(i))
    pprint(sent)
    r = ee.entities(sent)
    if 'entities' in r.json().keys():
        print('para {} entities are as follows:'.format(i))
        pprint(r.json())
        if len(r.json()['entities']) > 0:
            for entities in r.json()['entities']:
                para_level_entities += entities['count']

json_file = open(json_path)
json_str = json_file.read()
json_data = json.loads(json_str)[0]
doc_level_entities = 0
for entities in json_data['entities']:
    doc_level_entities += entities['count']


print('\033[95m' + ' Num Entities Detected' + '\033[95m')
print("When payload is DOCUMENT LEVEl : {}".format(doc_level_entities))
print("When payload is PARAGRAPH LEVEl : {}".format(para_level_entities))
