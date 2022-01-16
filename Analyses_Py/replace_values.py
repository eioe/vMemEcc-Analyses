"""Replaces the placeholders in the DOCX with the values from the JSON.

Arg 1: Path of DOCX
Arg 2: Path of JSON
Arg 3 (otional): Path of new file (default: <oldfile>_new.docx)
"""

import os
import shutil
import sys
import docxtpl
import json
import jinja2


path_file_in = sys.argv[1]
path_rep_dict = sys.argv[2]
if len(sys.argv) == 4:
    path_file_out = sys.argv[3]
else:
    path_file_out = path_file_in[:-5] + "_filledin.docx"

with open(path_rep_dict) as json_file:
    rep_dict = json.load(json_file)

path_file_copy = path_file_in[:-5] + "_tmp.docx"

# make copy to be on the safe side:
shutil.copyfile(path_file_in, path_file_copy)
docc = docxtpl.DocxTemplate(path_file_copy)
jinja_env = jinja2.Environment(undefined=jinja2.DebugUndefined)
# alternative: jinja2.StrictUndefined)
docc.render(rep_dict, jinja_env=jinja_env)

docc.save(path_file_out)


# clean up:
os.remove(path_file_copy)
