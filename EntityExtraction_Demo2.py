import requests
from pprint import pprint
import json
from nltk import  tokenize


class extract_entities(object):
    def __init__(self, text, rosette_key='03217af9eced40a5e2d1bcca4b76a4aa'):
        self.rosette = rosette_key
        self.headers = {'X-RosetteAPI-Key': self.rosette, 'Content-Type': 'application/json',
                        'Accept': 'application/json', 'Cache-Control': 'no-cache'}
        self.text = text

    def _word2offset(self):
        sent2words = {}
        for i, sent in enumerate(tokenize.sent_tokenize(self.text)):
            all_words = {}
            span_generator = tokenize.WhitespaceTokenizer().span_tokenize(sent)
            spans = [span for span in span_generator]
            words = tokenize.word_tokenize(sent)
            words = [word.lower() for word in words if word.isalpha()]
            for j, word in enumerate(words):
                word_props = {}
                word_props["span"] = spans[j]
                word_props["length"] = word.__len__()
                all_words[word] = word_props
            sent2words[sent] = all_words
        pprint(sent2words)

    def entities(self):
        data = {'content': self.text, 'genre': "regulatory", 'options': {"includeDBpediaType": True}}
        json_data = json.dumps(data)
        #print(json_data)
        response = requests.post('https://api.rosette.com/rest/v1/entities', headers=self.headers, data=json_data)
        return response

'''
Use example:
'''
etd_2 = "A valid currency of exchange between the banks in Netherlands, for e.g. ING and Rabobank, is Euro and not GBP. " \
        "Furthermore only transactions processed between 01-01-1990 and 31-12-2017 are valid. Any transaction in April is to be ignored." \
        "Transactions after May 1 2017 are valid again."

ee = extract_entities(etd_2)
ee._word2offset()
# r = ee.entities()
# pprint(r.json())
