import json, requests

PRODUCTION = True

class RosetteEntities:
    def __init__(self, rosette_key='03217af9eced40a5e2d1bcca4b76a4aa'):
        self.rosette = rosette_key
        self.headers = {'X-RosetteAPI-Key': self.rosette,
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                        'Cache-Control': 'no-cache'}

    def annotate(self, text):

        # max 50.000 chars per request
        lines = text.split("\n")
        req_text = ""
        result = []
        for line in lines:
            if len(req_text) + len(line) >= 49998:
                # post
                annotations = self.make_api_call(req_text)
                annotations['text'] = req_text
                result.append(annotations)
                req_text = line
            else:
                req_text = req_text + "\n" + line

        # add in the last section or if text didn't need to be split
        annotations = self.make_api_call(req_text)
        annotations['text'] = req_text
        result.append(annotations)

        return result

    def make_api_call(self, text):
        if not PRODUCTION:
            return {"entities": []}

        print("calling rosette api")
        options = {"includeDBpediaType": True,
                   "calculateConfidence": True,
                   "calculateSalience": True}
        data = {'content': text, 'genre': "regulatory", 'options': options}
        json_data = json.dumps(data)

        response = requests.post('https://api.rosette.com/rest/v1/entities', headers=self.headers, data=json_data)
        return response.json()
