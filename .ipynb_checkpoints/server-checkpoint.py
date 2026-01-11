#!/usr/bin/env python
import uvicorn
import argparse
import connexion
import os
from flask import send_from_directory, redirect, request
from flask_cors import CORS
from backend import AVAILABLE_MODELS

CONFIG_FILE_NAME = 'lmf.yml'
projects = {}

app = connexion.App(__name__)

class Project:
    """Wrapper around a model to allow lazy loading"""
    def __init__(self, model_cls, config_name):
        self.config = config_name
        self._model_cls = model_cls
        self._lm_instance = None

    @property
    def lm(self):
        """Lazy-load the model only once"""
        if self._lm_instance is None:
            print(f"Loading model '{self.config}' for the first time...")
            self._lm_instance = self._model_cls()
        return self._lm_instance

def get_all_projects():
    """Return configuration of all loaded projects"""
    return {k: projects[k].config for k in projects.keys()}

def analyze(analyze_request):
    """Analyze a text using the selected project/model"""
    project = analyze_request.get('project')
    text = analyze_request.get('text')
    topk = analyze_request.get('topk', 20)
    include_detectgpt = analyze_request.get('include_detectgpt', True)
    include_fastdetect = analyze_request.get('include_fastdetect', False)
    fastdetect_api_key = analyze_request.get('fastdetect_api_key')
    
    res = {}
    if project in projects:
        p = projects[project]
        lm_res = p.lm.check_probabilities(
            text, 
            topk=topk, 
            include_detectgpt=include_detectgpt,
            include_fastdetect=include_fastdetect,
            fastdetect_api_key=fastdetect_api_key
        )
        res["pred_topk"] = lm_res.get("pred_topk", [])
        res["real_topk"] = lm_res.get("real_topk", [])
        res["bpe_strings"] = lm_res.get("bpe_strings", [])
        res["detectgpt"] = lm_res.get("detectgpt", [])
        res["fastdetect"] = lm_res.get("fastdetect", {})
        
    return {
        "request": {
            'project': project, 
            'text': text, 
            'topk': topk, 
            'include_detectgpt': include_detectgpt,
            'include_fastdetect': include_fastdetect,
            'has_fastdetect_key': bool(fastdetect_api_key)  # Don't expose the actual key in response
        },
        "result": res
    }

#########################
#  Routes
#########################

@app.route('/')
def redir():
    """Redirect root to client UI"""
    proxy_prefix = request.script_root
    return redirect(f'{proxy_prefix}/client/index.html')

@app.route('/client/<path:path>')
def send_static(path):
    """Serve static files from client/dist"""
    return send_from_directory('client/dist/', path)

@app.route('/data/<path:path>')
def send_data(path):
    """Serve data files"""
    return send_from_directory(args.dir, path)

#########################
#  Main
#########################

app.add_api('server.yaml')

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='gpt-oss-20b')
parser.add_argument("--address", default="127.0.0.1")
parser.add_argument("--port", default="5001")
parser.add_argument("--dir", type=str, default=os.path.abspath('data'))
parser.add_argument("--no_cors", action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.no_cors:
        CORS(app.app, headers='Content-Type')

    app.run(port=int(args.port), host=args.address)
else:
    args, _ = parser.parse_known_args()
    # load_projects(args.dir)
    try:
        model = AVAILABLE_MODELS[args.model]
    except KeyError:
        model = AVAILABLE_MODELS['gpt-oss-20b','mbart-large-50']
    projects[args.model] = Project(model, args.model)
