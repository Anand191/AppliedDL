import string
import numpy as np
import spacy

nlp = spacy.load('en')

def additional_entities(span_arr):
    rows = np.where(span_arr[:, -1]=='OTHER')[0]
    for r in rows:
        if span_arr[r, -2].text in string.punctuation:
            span_arr[r, -1] = 'PUNCT'

        else:
            doc = nlp(span_arr[r, -2].text)
            try:
                if len(doc.ents) != 0:
                    for ent in doc.ents:
                        span_arr[r, -1] = ent.label_
            except:
                continue

    return span_arr


