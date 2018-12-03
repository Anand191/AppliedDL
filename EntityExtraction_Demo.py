import argparse
import json
import os
import pprint

from rosette.api import API, DocumentParameters, RosetteException


def run(key, alt_url='https://api.rosette.com/rest/v1/'):
    """ Run the example """
    # Create an API instance
    api = API(user_key=key, service_url=alt_url)

    etd_2 = "A valid currency of exchange between the banks in Netherlands, for e.g. ING and Rabobank, is Euro and not USD." \
            "Furthermore only transactions processed between 01-01-1990 and 31-12-2017 are valid."

    params = DocumentParameters()
    params["content"] = etd_2
    params["genre"] = "regulatory"

    try:
        return api.entities(params)
    except RosetteException as exception:
        print(exception)

PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Calls the ' +
                                 os.path.splitext(os.path.basename(__file__))[0] + ' endpoint')
PARSER.add_argument('-k', '--key', help='Rosette API Key', default = '03217af9eced40a5e2d1bcca4b76a4aa')
PARSER.add_argument('-u', '--url', help="Alternative API URL",
                    default='https://api.rosette.com/rest/v1/')

if __name__ == '__main__':
    ARGS = PARSER.parse_args()
    print(ARGS.key)
    RESULT = run(ARGS.key, ARGS.url)
    #print(json.dumps(RESULT, indent=2, ensure_ascii=False, sort_keys=True).encode("utf8"))
    pprint.pprint(RESULT)