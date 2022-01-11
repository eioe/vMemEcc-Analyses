import docx
import docxtpl
import json
import jinja2

#  TODO: make paths relative

path_rep_dict = 'C:\\Users\\Felix\\Seafile\\Experiments\\vMemEcc\\Writing\\Other\\VME_extracted_vars.json'
with open(path_rep_dict) as json_file:
    rep_dict = json.load(json_file)

docc = docxtpl.DocxTemplate('C:\\Users\\Felix\\Downloads\\test_rep_py.docx')
jinja_env = jinja2.Environment(undefined=jinja2.DebugUndefined) # jinja2.StrictUndefined)
docc.render(rep_dict, jinja_env=jinja_env)

docc.save('C:\\Users\\Felix\\Downloads\\test_rep_py_new.docx')

