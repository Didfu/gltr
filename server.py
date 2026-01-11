#!/usr/bin/env python
import uvicorn
import argparse
import connexion
import os
from flask import send_from_directory, redirect, request, send_file, jsonify
from flask_cors import CORS
from backend import AVAILABLE_MODELS
from dotenv import load_dotenv
from io import BytesIO
import json
from datetime import datetime
import pdfplumber
import logging
import tempfile
logging.getLogger("pdfminer").setLevel(logging.ERROR)
from connexion import request as connexion_request



# === PATH SETUP ===
# Only go one level up (project root = gltr/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dotenv_path = os.path.join(PROJECT_ROOT, ".env")

# Custom PDF save directory inside project root
pdf_dir = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(pdf_dir, exist_ok=True)

# Temporary PDF uploads directory
temp_pdf_dir = os.path.join(tempfile.gettempdir(), "scanit_uploads")
os.makedirs(temp_pdf_dir, exist_ok=True)


# === ENVIRONMENT ===
load_dotenv(dotenv_path)
key = os.getenv("key")

CONFIG_FILE_NAME = 'lmf.yml'
projects = {}


app = connexion.App(__name__)

# =====================================================================
# MODEL WRAPPER
# =====================================================================
class Project:
    """Wrapper around a model with pre-loading support"""
    def __init__(self, model_cls, config_name, preload=True):
        self.config = config_name
        self._model_cls = model_cls
        self._lm_instance = None
        
        # Pre-load if requested
        if preload:
            print(f"Pre-loading model '{self.config}'...")
            self._lm_instance = self._model_cls()
            print(f"Model '{self.config}' loaded successfully")

    @property
    def lm(self):
        """Return the model (should already be loaded)"""
        if self._lm_instance is None:
            raise RuntimeError(f"Model {self.config} was not pre-loaded")
        return self._lm_instance

def get_all_projects():
    """Return configuration of all loaded projects"""
    return {k: projects[k].config for k in projects.keys()}

# =====================================================================
# ANALYZE ENDPOINT
# =====================================================================
def analyze(analyze_request):
    """Analyze a text using the selected project/model"""
    print("=== ANALYZE FUNCTION CALLED ===")
    
    project = analyze_request.get('project')
    text = analyze_request.get('text')
    pdf_path = analyze_request.get('pdf_path')  # NEW: Optional PDF path
    
    print(f"Project requested: '{project}'")
    print(f"Available projects: {list(projects.keys())}")
    print(f"Text: '{text[:100]}...'")
    if pdf_path:
        print(f"PDF path provided: {pdf_path}")
    
    topk = analyze_request.get('topk', 20)
    include_detectgpt = analyze_request.get('include_detectgpt', True)
    include_fastdetect = analyze_request.get('include_fastdetect', True)
    include_factcheck = analyze_request.get('include_factcheck', True)
    max_claims = analyze_request.get('max_claims', 5)
    generate_gltr_viz = analyze_request.get('generate_gltr_viz', True)
    fastdetect_api_key = key
    
    res = {}
    
    try:
        if project in projects:
            print(f"✓ Project '{project}' found!")
            p = projects[project]
            
            if hasattr(p, 'lm'):
                print("Calling check_probabilities...")
                
                # Set PDF path if provided for GLTR overlay
                if pdf_path and os.path.exists(pdf_path):
                    print(f"Setting PDF path for GLTR overlay: {pdf_path}")
                    p.lm.set_current_pdf_path(pdf_path)
                
                lm_res = p.lm.check_probabilities(
                    text, 
                    topk=topk, 
                    include_detectgpt=include_detectgpt,
                    include_fastdetect=include_fastdetect,
                    fastdetect_api_key=fastdetect_api_key,
                    include_factcheck=include_factcheck,
                    max_claims=max_claims,
                    generate_gltr_viz=generate_gltr_viz,
                )
                
                print("✓ check_probabilities returned!")
                
                # Copy key results
                res["pred_topk"] = lm_res.get("pred_topk", [])
                res["real_topk"] = lm_res.get("real_topk", [])
                res["bpe_strings"] = lm_res.get("bpe_strings", [])
                res["detectgpt"] = lm_res.get("detectgpt", {})
                res["fastdetect"] = lm_res.get("fastdetect", {})
                res["factcheck"] = lm_res.get("factcheck", [])
                res["gltr_image"] = lm_res.get("gltr_image")

                # === Auto-generate PDF report ===
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pdf_filename = f"analysis_report_{timestamp}.pdf"
                    pdf_path_out = os.path.join(pdf_dir, pdf_filename)
                    
                    p.lm.generate_pdf_report(lm_res, output_path=pdf_path_out)
                    print(f"✓ PDF report saved: {pdf_path_out}")
                    res["pdf_filename"] = pdf_filename
                except Exception as e:
                    print(f"✗ Failed to generate PDF: {e}")
                    res["pdf_filename"] = None

                print(f"✓ Analysis complete. GLTR image: {res.get('gltr_image')}")
            else:
                print("✗ Project object has no 'lm' attribute!")
        else:
            print(f"✗ Project '{project}' NOT in projects dict!")
            
    except Exception as e:
        print(f"✗✗✗ EXCEPTION in analyze: {e}")
        import traceback
        traceback.print_exc()
        res["error"] = str(e)
    
    return {
        "request": {
            'project': project, 
            'text': text, 
            'topk': topk, 
            'include_detectgpt': include_detectgpt,
            'include_fastdetect': include_fastdetect,
            'include_factcheck': include_factcheck,
            'max_claims': max_claims,
            'generate_gltr_viz': generate_gltr_viz,
            'has_fastdetect_key': bool(fastdetect_api_key),
            'has_pdf_path': bool(pdf_path)
        },
        "result": res
    }

# =====================================================================
# MANUAL PDF REPORT GENERATION
# =====================================================================
def generate_report(report_request):
    """Manually generate PDF report from existing analysis"""
    print("=== GENERATE_REPORT FUNCTION CALLED ===")
    
    try:
        project = report_request.get('project')
        analysis_data = report_request.get('analysis_data')
        input_text = report_request.get('input_text', '')
        
        if not project or not analysis_data:
            return {"error": "Missing required fields: project and analysis_data"}, 400
        
        if project not in projects:
            return {"error": f"Project '{project}' not found"}, 404
        
        print(f"Generating PDF report for project: {project}")
        
        pdf_data = {
            "input_text": input_text,
            "bpe_strings": analysis_data.get("bpe_strings", []),
            "real_topk": analysis_data.get("real_topk", []),
            "pred_topk": analysis_data.get("pred_topk", []),
            "detectgpt": analysis_data.get("detectgpt"),
            "fastdetect": analysis_data.get("fastdetect"),
            "factcheck": analysis_data.get("factcheck", []),
            "gltr_image": analysis_data.get("gltr_image")
        }
        
        p = projects[project]
        pdf_buffer = p.lm.generate_pdf_report(pdf_data)
        
        print("✓ PDF report generated successfully")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"factcheck_report_{timestamp}.pdf"
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"✗ PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

# =====================================================================
# ROUTES
# =====================================================================
@app.route('/')
def serve_react():
    return send_from_directory('client/dist', 'index.html')

@app.route('/client/<path:path>')
def send_static(path):
    return send_from_directory('client', path)

@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory(args.dir, path)

@app.route('/api/health')
def health_check():
    return {
        "status": "healthy",
        "projects": list(projects.keys()),
        "models_loaded": len(projects)
    }

@app.route('/api/gltr_image/<path:filename>')
def serve_gltr_image(filename):
    import tempfile
    temp_dir = tempfile.gettempdir()
    return send_from_directory(temp_dir, filename)

@app.route('/api/download_pdf/<path:filename>')
def download_pdf(filename):
    """Serve previously generated PDF reports from project /reports folder"""
    file_path = os.path.join(pdf_dir, filename)
    if not os.path.exists(file_path):
        return {"error": f"File not found: {filename}"}, 404
    return send_from_directory(pdf_dir, filename, as_attachment=True, mimetype='application/pdf')

@app.route('/api/extract_pdf', methods=['POST'])
def extract_pdf():
    """Extract text from PDF and optionally save for GLTR overlay"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400
    
    try:
        # Save uploaded PDF temporarily for GLTR overlay
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"upload_{timestamp}_{file.filename}"
        temp_pdf_path = os.path.join(temp_pdf_dir, temp_filename)
        file.save(temp_pdf_path)
        print(f"Saved uploaded PDF to: {temp_pdf_path}")
        
        # Extract text from PDF
        text = ''
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text with layout preserved
                page_text = page.extract_text(layout=True)
                if page_text:
                    text += page_text + '\n\n'
        
        # Clean up common OCR artifacts
        text = text.replace(' - ', '-')  # Fix broken hyphens
        
        return jsonify({
            'text': text.strip(),
            'pdf_path': temp_pdf_path,  # Return path for GLTR overlay
            'pages': len(pdfplumber.open(temp_pdf_path).pages)
        })
    except Exception as e:
        print(f"PDF extraction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
def analyze_with_pdf(text, project='gemma-3n-E2B-it', file=None, topk=20,include_detectgpt=True, include_fastdetect=True, include_factcheck=True, max_claims=5, generate_gltr_viz=True):
    """
    Analyze text with optional PDF overlay for GLTR visualization
    Connexion automatically passes formData parameters as function arguments
    """
    print("=== ANALYZE_WITH_PDF FUNCTION CALLED ===")
    
    try:
        if not text:
            return {"error": "No text provided"}, 400
        
        print(f"Project: {project}")
        print(f"Text length: {len(text)}")
        print(f"File provided: {file is not None}")
        
        # Handle optional PDF file
        pdf_path = None
        if file is not None:
            # 'file' is a werkzeug FileStorage object
            if file.filename and file.filename.endswith('.pdf'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_filename = f"analyze_{timestamp}_{file.filename}"
                pdf_path = os.path.join(temp_pdf_dir, temp_filename)
                file.save(pdf_path)
                print(f"Saved PDF for analysis: {pdf_path}")
        
        # Build request dictionary for analyze function
        analyze_request = {
            'project': project,
            'text': text,
            'pdf_path': pdf_path,
            'topk': topk,
            'include_detectgpt': include_detectgpt,
            'include_fastdetect': include_fastdetect,
            'include_factcheck': include_factcheck,
            'max_claims': max_claims,
            'generate_gltr_viz': generate_gltr_viz
        }
        
        print(f"Calling analyze with project: {project}")
        print(f"Available projects: {list(projects.keys())}")
        
        # Call the analyze function
        result = analyze(analyze_request)
        
        return result
        
    except Exception as e:
        print(f"Error in analyze_with_pdf: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500
# =====================================================================
# CLEANUP ENDPOINT (Optional)
# =====================================================================
@app.route('/api/cleanup_temp_pdfs', methods=['POST'])
def cleanup_temp_pdfs():
    """Clean up old temporary PDF files (older than 1 hour)"""
    try:
        import time
        current_time = time.time()
        removed_count = 0
        
        for filename in os.listdir(temp_pdf_dir):
            file_path = os.path.join(temp_pdf_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                # Remove files older than 1 hour (3600 seconds)
                if file_age > 3600:
                    os.remove(file_path)
                    removed_count += 1
                    print(f"Removed old temp PDF: {filename}")
        
        return jsonify({
            'success': True,
            'removed_count': removed_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =====================================================================
# MAIN
# =====================================================================
app.add_api('server.yaml')

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='gemma-3n-E2B-it')
parser.add_argument("--address", default="127.0.0.1")
parser.add_argument("--port", default="5001")
parser.add_argument("--dir", type=str, default=os.path.abspath('data'))
parser.add_argument("--no_cors", action='store_true')

args, _ = parser.parse_known_args()

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.no_cors:
        CORS(app.app, headers='Content-Type')

    app.run(port=int(args.port), host=args.address)
else:
    args, _ = parser.parse_known_args()
    try:
        model = AVAILABLE_MODELS[args.model]
    except KeyError:
        model = AVAILABLE_MODELS['gemma-3n-E2B-it']
    projects[args.model] = Project(model, args.model)