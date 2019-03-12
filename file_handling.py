from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree
from bs4 import BeautifulSoup
import json


class FileHandler:
    def read_folder(self, folder_location):
        all_files = [f for f in listdir(folder_location) if isfile(join(folder_location, f))]
        return all_files

    def read_xml(self, folder_location, filename):
        file = open(join(folder_location, filename))
        e = xml.etree.ElementTree.parse(join(folder_location, filename))
        return e

    def html2text(self, html_file):
        soup = BeautifulSoup(html_file, 'html.parser')
        return "\n".join([x.get_text() for x in soup.find_all('p')])

    def write_json(self, data, folder_location, filename):
        with open(join(folder_location, "out", filename + ".json"), 'w') as outfile:
            json.dump(data, outfile)

    def read_json(self, folder_location, filename):
        with open(join(folder_location, filename)) as outfile:
            return json.load(outfile)
