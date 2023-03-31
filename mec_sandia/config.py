import os

SRC_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
REPO_DIRECTORY = "/" + "/".join(list(filter(None, SRC_DIRECTORY.split('/')))[:-1])
VASP_DATA = os.path.abspath(os.path.join(REPO_DIRECTORY, 'vasp_data'))