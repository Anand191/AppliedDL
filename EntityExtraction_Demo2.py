import requests
from pprint import pprint
import json

class extract_entities(object):
    def __init__(self, rosette_key='03217af9eced40a5e2d1bcca4b76a4aa'):
        self.rosette = rosette_key
        self.headers = {'X-RosetteAPI-Key': self.rosette, 'Content-Type': 'application/json',
                        'Accept': 'application/json', 'Cache-Control': 'no-cache'}

    def entities(self, text):
        data = {'content': text, 'genre': "regulatory", 'options': {"includeDBpediaType": True}}
        json_data = json.dumps(data)
        #print(json_data)
        response = requests.post('https://api.rosette.com/rest/v1/entities', headers=self.headers, data=json_data)
        return response

'''
Use example:
'''
etd_2 = "A valid currency of exchange between the banks in Netherlands, for e.g. ING and Rabobank, is Euro and not GBP. " \
        "Furthermore only transactions processed between 01-01-1990 and 31-12-2017 are valid."
ee = extract_entities()
r = ee.entities(etd_2)
pprint(r.json())



