import requests
from pprint import pprint
import json
from nltk import  tokenize

class extract_entities(object):
    def __init__(self, rosette_key='03217af9eced40a5e2d1bcca4b76a4aa'):
        self.rosette = rosette_key
        self.headers = {'X-RosetteAPI-Key': self.rosette, 'Content-Type': 'application/json',
                        'Accept': 'application/json', 'Cache-Control': 'no-cache'}

    def _word2offset(self, text):
        sent2words = {}
        for i, sent in enumerate(tokenize.sent_tokenize(text)):
            print(sent)
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
            pprint(all_words)

    def entities(self, text):
        data = {'content': text, 'genre': "regulatory", 'options': {"includeDBpediaType": True}}
        json_data = json.dumps(data)
        #print(json_data)
        response = requests.post('https://api.rosette.com/rest/v1/entities', headers=self.headers, data=json_data)
        return response