from os.path import join
import re, math, json

from file_handling import FileHandler
from ner import RosetteEntities

fh = FileHandler()
rosette = RosetteEntities()

FOLDER = 'resources/Test'

FORCE_RERUN = False

def main():
    files = fh.read_folder(FOLDER)
    # files = ['CELEX_32017R1991_EN_TXT.html']
    total_requests = 0
    total_htmls = 0
    for filename in files:

        # only process html files
        if not re.search(r'\.html?$', filename):
            continue

        document_name = re.search(r'([\_\-\w]+)\.html?', filename).group(1)

        if not FORCE_RERUN and document_name + ".json" in fh.read_folder(join(FOLDER, 'out')):
            print(document_name, "already has an output, not re-processing")
            continue

        # parse document en get NE's
        file = open(join(FOLDER, filename))
        text = fh.html2text(file)
        ners = rosette.annotate(text)

        # save as json
        fh.write_json(ners, FOLDER, document_name)

        # logging
        total_requests += math.ceil(len(text)/50000)
        total_htmls += 1
        print(filename, len(text), math.ceil(len(text)/50000))

    print("Total number of files:", total_htmls)
    print("Total number of requests:", total_requests)


def _clean_unwanted_chars(ne_set):
    # remove unicode character codes

    # go over the list backwards and update the NE's character indexes for each found unicode mention
    matches = [x for x in re.finditer(r'\\u[0-9,a-f]{4}', ne_set['text'])]
    for match in reversed(matches):
        # remove the unicode
        ne_set['text'] = ne_set['text'][:match.start()-1] + ne_set['text'][match.end():]

        # check and update each mention of each entity
        for entity in ne_set['entities']:
            for mention in entity['mentionOffsets']:
                # if the unicode occurred before the removed char, the mention indexes will need to be shifted
                if mention['startOffset'] >= match.end():
                    mention['startOffset'] -= 6
                    mention['endOffset'] -= 6

    return ne_set


def _split_sentences(ne_set):
    split = []
    prev_match = re.match(r'^', "this is to get a starting match object for the first sentence")
    matches = [x for x in re.finditer(r'(\n)+', ne_set['text'])]

    for match in re.finditer(r'(\n)+', ne_set['text']):

        sentence_start = prev_match.end()
        sentence_end = match.start()

        sentence = {
            'text': ne_set['text'][sentence_start:sentence_end],
            'entities': []
        }

        #
        for entity in ne_set['entities']:

            mentionOffsets = list(map(
                lambda entity_match: {
                    'startOffset': _char_pos_to_word_index(sentence['text'], entity_match.start()),
                    'endOffset': _char_pos_to_word_index(sentence['text'], entity_match.end())
                },
                re.finditer(entity['mention'], sentence['text'])
            ))

            if len(mentionOffsets) > 0:
                new_entity = {
                    'type': entity['type'],
                    'mention': entity['mention'],
                    'normalized': entity['normalized'],
                    'dbpediaType': entity['dbpediaType'] if 'dbpediaType' in entity else "Other/" + entity['type'],
                    'mentionOffsets': mentionOffsets
                }

                # print(new_entity)
                sentence['entities'].append(new_entity)

        split.append(sentence)
        prev_match = match

    return split


def _char_pos_to_word_index(text, start_char, end_char=False):
    char_counts = list(map(lambda word: len(word) + 1, text.split(" ")))

    start_word = -1
    end_word = -1

    char_sum = 0
    for i, word_length in enumerate(char_counts):
        if char_sum >= start_char:
            # found the start
            start_word = i
            break
        else:
            char_sum += word_length

    if end_char:
        for i, word_length in enumerate(list(char_counts)[start_word:]):
            char_sum += word_length
            if char_sum >= end_char:
                # found the end
                end_word = start_word + i
                break
    else:
        end_word = start_word

    return start_word


def restructure_entity_info(sentences):
    result = {
        "api_name": "rosette",
        "file_name": "example.pdf",
        "date_time": "",
        "sentences": []
    }

    for sentence in sentences:
        # print(sentence)
        entities = ""
        dbpedia_types = ""
        words = sentence['text'].split(" ")
        end = 0
        for i, word in enumerate(words):
            for entity in sentence['entities']:
                for entity_offset in entity['mentionOffsets']:
                    # print("e", entity)
                    if entity_offset['startOffset'] == end:
                        entities = " ".join([entities, "B-" + entity['type']])
                        end = entity_offset['startOffset'] + 1
                        if entity_offset['startOffset'] != entity_offset['endOffset']:
                            for j in range(entity_offset['endOffset'] - entity_offset['startOffset']):
                                entities = " ".join([entities, "I-" + entity['type']])
                                end = entity_offset['endOffset'] + 1
        if i > end:
            # none found
            entities = " ".join([entities, "O"])

        restructured_sentence = {
            "text": sentence['text'],
            "entities": entities,
            "dbpedia_types": dbpedia_types
        }
        result['sentences'].append(restructured_sentence)
    return result

def load_named_entities():
    files = fh.read_folder(join(FOLDER, 'out'))
    for filename in files:
        ne_sets = fh.read_json(join(FOLDER, 'out'), filename)

        for ne_set in ne_sets:
            cleaned = _clean_unwanted_chars(ne_set)
            print(cleaned)
            split = _split_sentences(cleaned)
            # print(split)
            restructured = restructure_entity_info(split)
            print(restructured)

if __name__ == '__main__':
    #main()
    load_named_entities()

