import json, re
from entity_drilldown import additional_entities

import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en')

filepath = 'resources/EC_regulations/out/CELEX_32013R1375_EN_TXT.json'
json_file = open(filepath)
json_str = json_file.read()
json_data = json.loads(json_str)[0]

rosette_entities = json_data['entities']

text = json_data['text']
text = re.sub('\n',' ', text)
doc = nlp(text)
#import pdb; pdb.set_trace()
sent_span = []
for sent in doc.sents:
    sent_span.append((sent.start_char, sent.end_char))

startOffset_tag = {}
endOffset_tag = {}
span_tags = []
for token in doc:
    if(token.text != " "):
        startOffset_tag[token.idx] = token
        endOffset_tag[token.idx+len(token.text)] = token
        span_tags.append([token.idx, token.idx+len(token.text), token])
        # print(token)
        # print("start_offset={}".format(token.idx), "end_offset={}".format(token.idx+len(token.text)))

span_tags = np.asanyarray(span_tags)
span_tags = np.c_[span_tags, np.zeros(span_tags.shape[0], dtype=object)]

for entity in rosette_entities:
    for offsets in entity['mentionOffsets']:
        rows = np.where(
            np.logical_and(
                np.greater_equal(span_tags[:,0],offsets['startOffset']),
                np.less_equal(span_tags[:,1], offsets['endOffset'])
            )
        )[0]
        for i, row in enumerate(rows):
            if(i == 0):
                span_tags[row, -1] = 'B-'+entity['type']
            else:
                span_tags[row, -1] = 'I-' + entity['type']

nent_rows = np.where(span_tags[:,-1]==0)[0]
span_tags[nent_rows,-1] = 'OTHER'
span_tags = additional_entities(span_tags)

seq2ent_arr = np.zeros((len(sent_span), 2), dtype=object)
for i, span in enumerate(sent_span):
    sent_rows = np.where(
            np.logical_and(
                np.greater_equal(span_tags[:,0],span[0]),
                np.less_equal(span_tags[:,1], span[1])
            )
        )[0]
    seq2ent_arr[i,0] = ' '.join(map(str, span_tags[sent_rows, -2]))
    seq2ent_arr[i,1] = ' '.join(map(str, span_tags[sent_rows, -1]))

df = pd.DataFrame(seq2ent_arr, columns=['Sequence', 'Entities'])
df.to_csv('./resources/EC_regulations/Seq2Ent_Data/CELEX_32013R1375_EN_TXT_new.csv')

