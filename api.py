import socket
import token
import numpy as np
import fitz  # PyMuPDF
import base64
import torch
import time
import os
import re
from dotenv import load_dotenv
import traceback
from accelerate import init_empty_weights, infer_auto_device_map
import requests
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mxfp4Config, T5Tokenizer, T5ForConditionalGeneration
from urllib.parse import quote_plus
from typing import List, Dict, Any
from difflib import SequenceMatcher
import asyncio
import requests.packages.urllib3.util.connection as urllib3_cn
from transformers import BitsAndBytesConfig
import sys
from PIL import Image as PILImage
from reportlab.platypus import Image, Spacer, PageBreak
from reportlab.lib.units import inch
from PIL import Image as PILImage
# Add these to your imports at the top of api.py
from io import BytesIO
import tempfile
from datetime import datetime
import math
from PIL import Image as PILImage
import base64, tempfile, time, traceback
from io import BytesIO
import fitz
import asyncio
from playwright.async_api import async_playwright
# For PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, 
    HRFlowable, KeepTogether, CondPageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from PIL import Image as PILImage
import sys, importlib, socket
import re
import unicodedata

if not hasattr(socket, "AF_INET"):
    print("⚠️ socket module corrupted, reloading real one...")
    socket = importlib.import_module("socket")
    sys.modules["socket"] = socket

# For GLTR visualization with Pyppeteer
import asyncio
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    print("Warning: playwright not installed. Install with: pip install playwright")
    PLAYWRIGHT_AVAILABLE = False

from .class_register import register_api

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(PROJECT_ROOT, ".env")

load_dotenv(dotenv_path)
auth = os.getenv("auth")
key = os.getenv("key")

# ================= FACTCHECK CONFIG =================
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX")



def preprocess_pdf_text(raw_text: str) -> str:
    """
    Cleans raw PDF text before passing to the analysis pipeline.
    Removes excessive whitespace, page headers/footers, and junk chars
    while preserving structure and readability.
    """

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", raw_text)

    #  Remove control characters & non-printables
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

    # Remove PDF headers/footers (common)
    text = re.sub(r'Page\s*No\s*\d+\s*Department.*', '', text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r' *\n *', '\n', text)

    # Merge split lines that continue sentences
    text = re.sub(r'(?<![.:\-])\n(?!\n)', ' ', text)

    # Preserve paragraph breaks
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Simplify punctuation and artifacts
    text = re.sub(r'([.,;:!?]){2,}', r'\1', text)
    text = re.sub(r'[\-–_]{3,}', '—', text)

    # Normalize spacing around punctuation
    text = re.sub(r'\s*([:/])\s*', r'\1 ', text)

    # Collapse long underscore/dot sequences (tables, dividers)
    text = re.sub(r'[\._]{4,}', ' ', text)

    # Remove short meaningless lines (optional)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 2:
            lines.append(line)
    text = "\n".join(lines).strip()

    return text
# ================= FACTCHECK UTILS =================
def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower()).strip()

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

class AbstractLanguageChecker:
    """
    Abstract Class that defines the Backend API of GLTR.

    To extend the GLTR interface, you need to inherit this and
    fill in the defined functions.
    """

    def __init__(self):
        """
        In the subclass, you need to load all necessary components
        for the other functions.
        Typically, this will comprise a tokenizer and a model.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        

    def check_probabilities(self, in_text, topk=40):
        """
        Function that GLTR interacts with to check the probabilities of words

        Params:
        - in_text: str -- The text that you want to check
        - topk: int -- Your desired truncation of the head of the distribution

        Output:
        - payload: dict -- The wrapper for results in this function, described below

        Payload values
        ==============
        bpe_strings: list of str -- Each individual token in the text
        real_topk: list of tuples -- (ranking, prob) of each token
        pred_topk: list of list of tuple -- (word, prob) for all topk
        """
        raise NotImplementedError

    def postprocess(self, token):
    # Handle torch.Tensor or numpy.int
        if hasattr(token, "item"):
            token = token.item()
        if isinstance(token, str):
            return token    

    # If it's a single int → wrap in list
        if isinstance(token, int):
            return self.tokenizer.decode([token], clean_up_tokenization_spaces=True)

    # If it's already a list/tuple/array of ints
        if isinstance(token, (list, tuple)):
            return self.tokenizer.decode(token, clean_up_tokenization_spaces=True)

        raise TypeError(f"Unsupported token type: {type(token)}")




    def top_k_logits(logits, k):
     """
    Filters logits to only the top k choices
    from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
    """
     if k == 0:
        return logits
     values, _ = torch.topk(logits, k)
     min_values = values[:, -1]
     return torch.where(logits < min_values,torch.ones_like(logits, torch_dtype=logits.dtype) * -1e10,logits)

_MODULE_INITIALIZED = False
@register_api(name='gemma-3-270m-it')
class Gemma2BItChecker:
    cache_dir=r"C:\Users\dhruv mahyavanshi\.cache\huggingface\hub"
    os.environ["XDG_CACHE_HOME"] = cache_dir
    EXTRA_ID_PATTERN = re.compile(r"<extra_id_\d+>")
    _models_loaded = False
    _shared_model = None
    _shared_tokenizer = None
    _shared_mask_model = None
    _shared_mask_tokenizer = None
    _shared_device = None
    def __init__(self,
                base_model_name="google/gemma-3-270m-it",
                mask_filling_model_name="google-t5/t5-small",
                hfauth=auth,
                topk=40,
                int8=False,
                half=False,
                base_half=False):

        # ✅ Prevent multiple loads (shared class cache)
        if Gemma2BItChecker._models_loaded:
            print("✅ Models already loaded — reusing existing models.")
            self.model = Gemma2BItChecker._shared_model
            self.tokenizer = Gemma2BItChecker._shared_tokenizer
            self.mask_model = Gemma2BItChecker._shared_mask_model
            self.mask_tokenizer = Gemma2BItChecker._shared_mask_tokenizer
            self.device = Gemma2BItChecker._shared_device
            self.topk = topk
            self.base_model_name = base_model_name
            self.mask_filling_model_name = mask_filling_model_name
            return

        # ============ DEVICE SELECTION ============ #
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'✓ Using GPU: {torch.cuda.get_device_name(0)}')
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'✓ VRAM Available: {vram_gb:.2f} GB')
        else:
            self.device = torch.device("cpu")
            print('⚠ CUDA not available, using CPU')

        self.topk = topk
        self.base_model_name = base_model_name
        self.mask_filling_model_name = mask_filling_model_name
        self.cache_dir = getattr(self, "cache_dir", os.path.join(os.getcwd(), "model_cache"))
        os.makedirs(self.cache_dir, exist_ok=True)

        # ============ LOAD BASE MODEL (Gemma) ============ #
        print(f'Loading BASE model {base_model_name}...')

        if self.device.type == "cuda":
            print("→ Using 4-bit quantized loading for Gemma (VRAM-efficient)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
                token=hfauth,
                cache_dir=self.cache_dir
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                token=hfauth,
                cache_dir=self.cache_dir
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            token=hfauth,
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        torch.set_grad_enabled(False)
        self.model.config.use_cache = False
        print(f'✓ Base model loaded (4-bit) on {self.device}')

        # GPU memory check
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            mem_used = torch.cuda.memory_allocated(0) / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            mem_free = mem_total - mem_used
            print(f'📊 After base model - GPU Memory: {mem_used:.2f} GB / {mem_total:.2f} GB (Free: {mem_free:.2f} GB)')
        else:
            mem_free = None

        # ============ LOAD MASK FILLING MODEL (T5-small) ============ #
        if self.device.type == "cuda" and (mem_free is None or mem_free >= 0.5):
            print("✓ Loading mask model on GPU (4-bit quantized)")
            bnb_mask_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.mask_model = T5ForConditionalGeneration.from_pretrained(
                mask_filling_model_name,
                device_map="auto",
                quantization_config=bnb_mask_config,
                cache_dir=self.cache_dir
            )
            mask_device = self.device
        else:
            print("⚠️  Loading mask model on CPU to save VRAM")
            self.mask_model = T5ForConditionalGeneration.from_pretrained(
                mask_filling_model_name,
                torch_dtype=torch.float32,
                cache_dir=self.cache_dir
            )
            mask_device = torch.device("cpu")

        self.mask_tokenizer = T5Tokenizer.from_pretrained(
            mask_filling_model_name,
            cache_dir=self.cache_dir,
            model_max_length=512
        )

        self.mask_model.eval()
        torch.set_grad_enabled(False)
        print(f'✓ Mask filling model loaded on {mask_device}')

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            mem_used = torch.cuda.memory_allocated(0) / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'✓ Final GPU Memory Used: {mem_used:.2f} GB / {mem_total:.2f} GB')

        print(f'✅ All models ready - Base: {self.device}, Mask: {mask_device}')

        # ============ SHARE MODELS ============ #
        Gemma2BItChecker._shared_model = self.model
        Gemma2BItChecker._shared_tokenizer = self.tokenizer
        Gemma2BItChecker._shared_mask_model = self.mask_model
        Gemma2BItChecker._shared_mask_tokenizer = self.mask_tokenizer
        Gemma2BItChecker._shared_device = self.device
        Gemma2BItChecker._models_loaded = True

    async def _generate_gltr_image_async(self, gltr_data: dict, output_path: str = None) -> str:
        """
        Generate high-resolution GLTR visualization using Playwright with multi-page support
        """
        if not PLAYWRIGHT_AVAILABLE:
            print("Playwright not available, skipping GLTR image generation")
            return None
            
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"gltr_{int(time.time())}.png")
        
        try:
            print("Launching headless browser for GLTR visualization...")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                
                # Create HTML with GLTR visualization
                html_content = self._create_gltr_html(gltr_data)
                
                # New: use temporary page first to calculate dynamic height
                temp_context = await browser.new_context()
                temp_page = await temp_context.new_page()
                await temp_page.set_content(html_content)
                content_height = await temp_page.evaluate("document.body.scrollHeight")
                await temp_context.close()

                # ✅ Dynamic viewport size based on content height
                viewport_width = 1600  # wider for better scaling
                viewport_height = min(9000, int(content_height * 1.1))  # 10% padding, cap at 9k

                print(f"Dynamic viewport set → width={viewport_width}px, height={viewport_height}px (content={content_height}px)")

                page = await browser.new_page(viewport={'width': viewport_width, 'height': viewport_height})
                await page.set_content(html_content)

                # Give time for rendering
                await asyncio.sleep(1.5)

                # ✅ Scale and style improvements for better page appearance
                await page.evaluate("""
                    document.body.style.background = '#f4f4f6';
                    document.body.style.padding = '60px 40px';
                    document.querySelectorAll('.text-container').forEach(el => {
                        el.style.padding = '60px 80px';
                        el.style.marginBottom = '80px';
                        el.style.borderRadius = '16px';
                        el.style.boxShadow = '0 6px 20px rgba(0,0,0,0.1)';
                    });
                    document.querySelectorAll('.token').forEach(el => {
                        el.style.transform = 'scale(1.05)';
                        el.style.display = 'inline-block';
                        el.style.margin = '0 2px';
                    });
                """)

                # Capture screenshot
                await page.screenshot(path=output_path, full_page=True)
                await browser.close()
                print(f"✓ GLTR visualization saved: {output_path}")
                return output_path

        except Exception as e:
            print(f"✗ Playwright error: {e}")
            traceback.print_exc()
            return None

    def _create_gltr_html(self, gltr_data: dict) -> str:
        """
        Create HTML string for GLTR visualization with page break support
        """
        bpe_strings = gltr_data.get("bpe_strings", [])
        real_topk = gltr_data.get("real_topk", [])
        
        def get_color(topk_pos):
            """Match colors from your start.js"""
            if topk_pos < 10:
                return '#ADFF80'  # Green
            elif topk_pos < 100:
                return '#FFEA80'  # Yellow
            elif topk_pos < 1000:
                return '#FF9280'  # Red
            else:
                return '#E5B0FF'  # Purple
        
        def clean_token(token):
            """Clean token for display while preserving spaces"""
            if token in ['<bos>', '</s>', '<pad>']:
                return f'⟨{token[1:-1]}⟩'
            if token == '⟨bos⟩':
                return token
            return token
        
        # Generate token HTML with page break detection
        tokens_html = []
        chars_in_current_page = 0
        max_chars_per_page = 3000  # Approximate characters per visual "page"
        
        for i in range(len(bpe_strings) - 1):
            token = bpe_strings[i + 1]
            topk_pos, prob = real_topk[i]
            color = get_color(topk_pos)
            cleaned = clean_token(token)
            
            # Check if token starts with space indicator (▁ or Ġ for different tokenizers)
            has_leading_space = cleaned.startswith('▁') or cleaned.startswith('Ġ')
            
            # Remove the space indicator for display
            if has_leading_space:
                display_token = cleaned[1:]  # Remove first character (▁ or Ġ)
            else:
                display_token = cleaned
            
            # Escape HTML characters
            display_token = display_token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Skip empty tokens
            if not display_token.strip():
                continue
            
            # Check if we need a page break
            chars_in_current_page += len(display_token)
            if chars_in_current_page > max_chars_per_page and has_leading_space:
                # Add page break
                tokens_html.append('<div class="page-break"></div>')
                chars_in_current_page = 0
            
            # Add actual space before token if it had the space indicator
            if has_leading_space and len(tokens_html) > 0:
                # Add a space as text, not in a span
                tokens_html.append(' ')
            
            # Add the token span
            tokens_html.append(
                f'<span class="token" style="background-color: {color};" '
                f'data-topk="{topk_pos}" data-prob="{prob:.4f}">{display_token}</span>'
            )
        
        return f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 
                            'Helvetica Neue', Arial, sans-serif;
                background: #e5e5e5;
                padding: 40px 20px;
            }}
            
            .container {{
                max-width: 1300px;
                margin: 0 auto;
            }}
            
            .header {{
                background: white;
                padding: 40px;
                margin-bottom: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            h1 {{
                color: #1a237e;
                font-size: 32px;
                font-weight: 600;
                margin-bottom: 25px;
                text-align: center;
                letter-spacing: -0.5px;
            }}
            
            .legend {{
                display: flex;
                justify-content: center;
                gap: 40px;
                flex-wrap: wrap;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 15px;
                color: #424242;
                font-weight: 500;
            }}
            
            .legend-box {{
                width: 24px;
                height: 24px;
                border-radius: 4px;
                border: 1px solid rgba(0, 0, 0, 0.1);
                flex-shrink: 0;
            }}

            .legend-green {{ background-color: #10b981; }}
            .legend-yellow {{ background-color: #fbbf24; }}
            .legend-red {{ background-color: #ef4444; }}
            .legend-purple {{ background-color: #8b5cf6; }}
            
            /* Page-like content container */
            .text-container {{
                background: white;
                padding: 50px 60px;
                margin-bottom: 20px;
                border-radius: 10px;
                border: 1px solid #d0d0d0;
                line-height: 2.2;
                font-size: 17px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                white-space: normal;  /* Changed from pre-wrap to normal */
                word-wrap: break-word;
                min-height: 800px;
            }}
            
            /* Page break styling */
            .page-break {{
                display: block;
                height: 60px;
                background: linear-gradient(to bottom, 
                    rgba(255,255,255,1) 0%, 
                    rgba(229,229,229,1) 50%, 
                    rgba(255,255,255,1) 100%);
                margin: 40px -60px;
                border-top: 2px dashed #ccc;
                border-bottom: 2px dashed #ccc;
                position: relative;
            }}
            
            .page-break::after {{
                content: "--- Page Break ---";
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #999;
                font-size: 12px;
                font-weight: 600;
                background: white;
                padding: 5px 15px;
                border-radius: 4px;
            }}
            
            .token {{
                display: inline;  /* Keep inline for natural text flow */
                padding: 2px 4px;
                margin: 0;
                border-radius: 3px;
                cursor: default;
                transition: all 0.2s;
                border: 1px solid transparent;
                font-size: 10pt;
                font-weight: 500;
                white-space: normal;  /* Changed from pre-wrap */
            }}

            .token:hover {{
                box-shadow: inset 0 0 3px #333f50;
            }}

            .stats {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 20px;
            }}
            
            .stat-item {{
                text-align: center;
            }}
            
            .stat-value {{
                font-size: 28px;
                font-weight: 700;
                color: #1a237e;
            }}
            
            .stat-label {{
                font-size: 14px;
                color: #666;
                margin-top: 8px;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="text-container">
                {''.join(tokens_html)}
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">{len(bpe_strings) - 1}</div>
                    <div class="stat-label">Total Tokens</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{sum(1 for _, (pos, _) in enumerate(real_topk) if pos < 10)}</div>
                    <div class="stat-label">Top-10 Tokens</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{sum(1 for _, (pos, _) in enumerate(real_topk) if pos >= 1000)}</div>
                    <div class="stat-label">Beyond Top-1000</div>
                </div>
            </div>
        </div>
    </body>
    </html>"""
    
    async def _generate_gltr_pdf_overlay_screenshot_async(self, gltr_data: dict, original_pdf_path: str, output_path: str = None) -> str:
        """
        Generate GLTR visualization using PDF -> Images -> HTML -> Screenshot pipeline
        with dynamic spacing and scaling based on total page count
        """
        if not PLAYWRIGHT_AVAILABLE:
            print("Playwright not available")
            return None

        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"gltr_overlay_{int(time.time())}.png")

        try:
            pdf_document = fitz.open(original_pdf_path)
            total_pages = len(pdf_document)

            # Dynamically adjust spacing and image scale
            base_margin = 40
            margin_scale = min(80, base_margin + total_pages * 8)  # up to +80px margin for long docs
            img_padding = max(20, 60 - total_pages * 2)             # reduce inner padding slightly if too many pages
            zoom_factor = 2.0 if total_pages < 5 else 1.5 if total_pages < 10 else 1.25

            print(f"Detected {total_pages} pages → using zoom={zoom_factor}, margin={margin_scale}px, padding={img_padding}px")

            page_images = []
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = PILImage.open(BytesIO(img_data))

                # Add GLTR overlay (from your existing method)
                img_with_overlay = self._add_gltr_overlay_to_image(img, page, gltr_data, page_num)

                buffered = BytesIO()
                img_with_overlay.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                page_images.append(img_base64)

            pdf_document.close()

            # Combine all images into HTML dynamically spaced
            pages_html = ""
            for idx, img_base64 in enumerate(page_images):
                pages_html += f'''
                    <div class="pdf-page">
                        <div class="page-number">Page {idx + 1}</div>
                        <img src="data:image/png;base64,{img_base64}" alt="Page {idx + 1}" />
                    </div>
                '''

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        background: #f4f4f6;
                        padding: {margin_scale}px 30px;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    .header {{
                        background: white;
                        padding: 50px 60px;
                        margin-bottom: {margin_scale}px;
                        border-radius: 16px;
                        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        text-align: center;
                        color: #1a237e;
                        font-size: 32px;
                        font-weight: 600;
                        margin-bottom: 25px;
                    }}
                    .legend {{
                        display: flex;
                        justify-content: center;
                        gap: 40px;
                        flex-wrap: wrap;
                    }}
                    .legend-item {{
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        font-size: 15px;
                        color: #333;
                        font-weight: 500;
                    }}
                    .legend-box {{
                        width: 24px;
                        height: 24px;
                        border-radius: 4px;
                        border: 1px solid rgba(0,0,0,0.1);
                    }}
                    .legend-green {{ background-color: rgba(173, 255, 128, 0.6); }}
                    .legend-yellow {{ background-color: rgba(255, 234, 128, 0.6); }}
                    .legend-red {{ background-color: rgba(255, 146, 128, 0.6); }}
                    .legend-purple {{ background-color: rgba(229, 176, 255, 0.6); }}
                    .pdf-page {{
                        background: white;
                        padding: {img_padding}px;
                        margin-bottom: {math.ceil(margin_scale/1.2)}px;
                        border-radius: 14px;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
                        position: relative;
                    }}
                    .page-number {{
                        text-align: center;
                        color: #555;
                        font-size: 15px;
                        margin-bottom: 15px;
                        font-weight: 600;
                    }}
                    .pdf-page img {{
                        width: 100%;
                        display: block;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    {pages_html}
                </div>
            </body>
            </html>
            """

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={'width': 1400, 'height': int(2000 + total_pages * 100)})
                await page.set_content(html_content)
                await asyncio.sleep(1.5)
                await page.screenshot(path=output_path, full_page=True)
                await browser.close()

            print(f"✓ GLTR PDF overlay screenshot saved dynamically: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Error: {e}")
            traceback.print_exc()
            return None

    def _add_gltr_overlay_to_image(self, img: 'PILImage', page, gltr_data: dict, page_num: int) -> 'PILImage':
        """Add semi-transparent GLTR color overlays to PDF page image"""
        from PIL import ImageDraw
        
        # Create overlay layer
        overlay = PILImage.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Get text positions from page
        text_instances = page.get_text("words")
        bpe_strings = gltr_data.get("bpe_strings", [])
        real_topk = gltr_data.get("real_topk", [])
        
        # Build full text from tokens
        full_text = ""
        token_positions = []
        char_pos = 0
        
        for i in range(len(bpe_strings) - 1):
            token = bpe_strings[i + 1]
            topk_pos, prob = real_topk[i]
            
            # Handle space indicator
            if token.startswith('▁') or token.startswith('Ġ'):
                token_text = ' ' + token[1:]
            else:
                token_text = token
            
            # Skip special tokens
            if token in ['<bos>', '</s>', '<pad>', '⟨bos⟩']:
                continue
            
            token_positions.append({
                'text': token_text,
                'start': char_pos,
                'end': char_pos + len(token_text),
                'topk': topk_pos,
                'prob': prob
            })
            
            full_text += token_text
            char_pos += len(token_text)
        
        # Match words from PDF to tokens
        pdf_text = page.get_text()
        word_char_pos = 0
        
        for word_info in text_instances:
            x0, y0, x1, y1, word = word_info[:5]
            
            # Find word position in reconstructed text
            word_start = pdf_text.find(word, word_char_pos)
            if word_start == -1:
                continue
            word_end = word_start + len(word)
            word_char_pos = word_end
            
            # Find overlapping tokens
            for token_info in token_positions:
                token_overlap_start = max(token_info['start'], word_start)
                token_overlap_end = min(token_info['end'], word_end)
                
                if token_overlap_start < token_overlap_end:
                    # This token overlaps with this word - color it
                    topk_pos = token_info['topk']
                    
                    # Get color based on topk position
                    if topk_pos < 10:
                        color = (173, 255, 128, 100)  # Green
                    elif topk_pos < 100:
                        color = (255, 234, 128, 100)  # Yellow
                    elif topk_pos < 1000:
                        color = (255, 146, 128, 100)  # Red
                    else:
                        color = (229, 176, 255, 100)  # Purple
                    
                    # Scale coordinates (2x zoom from matrix)
                    draw.rectangle([x0*2, y0*2, x1*2, y1*2], fill=color)
                    break  # Only apply first matching token
        
        # Composite overlay onto original image
        img = img.convert('RGBA')
        img = PILImage.alpha_composite(img, overlay)
        
        return img

    def generate_gltr_pdf_screenshot(self, gltr_data: dict, original_pdf_path: str, output_path: str = None) -> str:
        """
        Synchronous wrapper for GLTR PDF screenshot generation
        
        Args:
            gltr_data: Dictionary with bpe_strings and real_topk
            original_pdf_path: Path to original PDF file
            output_path: Output PNG path
        
        Returns:
            Path to generated image or None on failure
        """
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                self._generate_gltr_pdf_overlay_screenshot_async(gltr_data, original_pdf_path, output_path)
            )
        except Exception as e:
            print(f"Failed to generate GLTR PDF screenshot: {e}")
            return None

    def generate_gltr_image(self, gltr_data: dict, output_path: str = None) -> str:
        """
        Synchronous wrapper for GLTR image generation
        
        Args:
            gltr_data: Dictionary with bpe_strings and real_topk
            output_path: Output PNG path
        
        Returns:
            Path to generated image or None on failure
        """
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                self._generate_gltr_image_async(gltr_data, output_path)
            )
        except Exception as e:
            print(f"Failed to generate GLTR image: {e}")
            return None
        
    # ================= FACTCHECK METHODS =================
    def generate_claims(self, text: str, max_claims: int = 5) -> List[str]:
        """
        Generate structured, analytical claims from text using iterative refinement.
        Prioritizes temporal, statistical, and verifiable factual statements.
        """
        print(f"Generating analytical claims from text...")
        
        # Step 1: Extract key sentences that contain factual statements
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 10]
        
        if not sentences:
            print("No valid sentences found in text")
            return []
        
        candidate_claims = []
        
        for sentence in sentences[:max_claims * 3]:  # Process more to filter down
            # Clean the sentence
            clean_sentence = sentence.strip()
            if len(clean_sentence) < 20 or len(clean_sentence) > 300:
                continue
                
            # Skip sentences that are too vague or incomplete
            if any(phrase in clean_sentence.lower() for phrase in ['which will', 'thus we can', 'for example', 'such as']):
                if len(clean_sentence) < 50:  # Too short for context phrases
                    continue
            
            # Create a structured prompt for claim generation
            prompt = f"Rewrite as a complete factual claim: {clean_sentence}"
            
            try:
                inputs = self.mask_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=256, 
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.mask_model.generate(
                        **inputs,
                        max_length=100,
                        min_length=15,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.6,  # Lower temperature for more focused output
                        top_p=0.85,
                        top_k=40,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.2,
                        early_stopping=True
                    )
                
                generated_text = self.mask_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the generated claim
                claim = self._clean_generated_claim(generated_text, clean_sentence)
                
                # Score the claim quality
                if claim:
                    quality_score = self._score_claim_quality(claim)
                    
                    if quality_score >= 0.6:  # Only accept high-quality claims
                        candidate_claims.append({
                            'text': claim,
                            'score': quality_score,
                            'original': clean_sentence
                        })
                        print(f"  ✓ Candidate claim (score: {quality_score:.2f}): {claim[:80]}...")
                    else:
                        print(f"  ✗ Low quality (score: {quality_score:.2f}): {claim[:60]}...")
                        
            except Exception as e:
                print(f"  ✗ Error generating claim from sentence: {e}")
                continue
        
        # Sort by quality score and take top claims
        candidate_claims.sort(key=lambda x: x['score'], reverse=True)
        claims = [c['text'] for c in candidate_claims[:max_claims]]
        
        # Fallback: If not enough high-quality claims, extract directly
        if len(claims) < max_claims:
            print(f"Only {len(claims)} high-quality claims found, extracting more directly...")
            direct_claims = self._extract_claims_directly(text, max_claims - len(claims))
            claims.extend(direct_claims)
        
        print(f"Successfully generated {len(claims)} analytical claims")
        return claims

    def _score_claim_quality(self, claim: str) -> float:
        """
        Score claim quality from 0-1 based on multiple factors.
        Higher score = better claim for fact-checking.
        Prioritizes temporal, statistical, and specific data.
        """
        score = 0.0
        words = claim.split()
        
        # 1. Length score (20-150 words is ideal)
        word_count = len(words)
        if 20 <= word_count <= 150:
            score += 0.3
        elif 10 <= word_count < 20 or 150 < word_count <= 200:
            score += 0.15
        else:
            score += 0.0
        
        # 2. Completeness score
        # Should end with proper punctuation
        if claim.strip()[-1] in '.!':
            score += 0.1
        
        # Should not end with incomplete phrases
        incomplete_endings = ['...', 'etc', 'and so on', 'or', 'and', 'but', 'which', 'that']
        if not any(claim.lower().strip().endswith(ending) for ending in incomplete_endings):
            score += 0.1
        
        # Should not start with connectives (fragment indicator)
        fragment_starters = ['which', 'that', 'thus', 'therefore', 'however', 'although', 'because']
        if not any(claim.lower().strip().startswith(starter) for starter in fragment_starters):
            score += 0.1
        
        # 3. Content quality - ENHANCED FOR TEMPORAL AND SPECIFIC DATA
        # Has numbers (general)
        has_numbers = bool(re.search(r'\d+', claim))
        if has_numbers:
            score += 0.1
        
        # Has specific dates (year, month-year, full date) - HIGHLY VALUABLE
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Year (1900-2099)
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}\b',  # Month Day, Year
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+(19|20)\d{2}\b',  # Abbreviated month
            r'\b\d{1,2}/\d{1,2}/(19|20)?\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(19|20)\d{2}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\bin\s+(19|20)\d{2}\b',  # "in 2024"
            r'\bsince\s+(19|20)\d{2}\b',  # "since 2020"
            r'\buntil\s+(19|20)\d{2}\b',  # "until 2025"
        ]
        
        has_date = any(re.search(pattern, claim, re.IGNORECASE) for pattern in date_patterns)
        if has_date:
            score += 0.2  # Extra bonus for dates
            print(f"    [DATE DETECTED] +0.2 bonus")
        
        # Has temporal language (published, announced, released, etc.)
        temporal_indicators = [
            r'\b(published|released|announced|launched|introduced|revealed|disclosed)\b',
            r'\b(reported|stated|confirmed|declared|proclaimed)\b',
            r'\b(occurred|happened|took place|began|started|ended|concluded)\b',
            r'\b(founded|established|created|formed)\b',
            r'\b(updated|revised|amended|modified)\b'
        ]
        
        has_temporal = any(re.search(pattern, claim, re.IGNORECASE) for pattern in temporal_indicators)
        if has_temporal:
            score += 0.15
            print(f"    [TEMPORAL ACTION] +0.15 bonus")
        
        # Percentage/statistics (very verifiable)
        has_percentage = bool(re.search(r'\d+%|\d+\s*percent', claim, re.IGNORECASE))
        if has_percentage:
            score += 0.15
            print(f"    [PERCENTAGE] +0.15 bonus")
        
        # Specific measurements/quantities
        measurement_patterns = [
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',  # Money
            r'\d+\s*(?:km|miles|meters|feet|kg|pounds|tons|liters|gallons)',  # Measurements
            r'\d+\s*(?:people|users|customers|employees|students|deaths|cases)',  # Counts
        ]
        
        has_measurement = any(re.search(pattern, claim, re.IGNORECASE) for pattern in measurement_patterns)
        if has_measurement:
            score += 0.1
            print(f"    [MEASUREMENT] +0.1 bonus")
        
        # Has proper nouns (names, organizations, places)
        has_capitalized = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim))
        if has_capitalized:
            score += 0.1
        
        # 4. Grammatical structure
        # Should have a verb
        verb_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'can', 'will', 
                        'would', 'could', 'should', 'may', 'might', 'must', 'do', 'does', 'did',
                        'provides', 'shows', 'indicates', 'demonstrates', 'reveals',
                        'published', 'announced', 'released', 'reported', 'stated']
        verb_count = sum(1 for verb in verb_indicators if f' {verb} ' in f' {claim.lower()} ')
        if verb_count > 0:
            score += 0.15
        
        # 5. Penalty for bad patterns
        bad_patterns = [
            r'<extra_id',
            r'<pad>',
            r'\.\.\.',
            r'A word is not',
            r'Which will provide',
            r'^[A-Z]\s+word\s+',  # "A word is..."
            r'\s+\.\s+\.',  # Double periods
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, claim):
                score -= 0.5
                break
        
        return max(0.0, min(1.0, score))

    def _is_valid_claim(self, claim: str) -> bool:
        """
        Validate that a claim is factual and check-worthy.
        """
        # Length check
        if len(claim) < 20 or len(claim) > 500:
            return False
        
        # Must have some content words
        words = claim.split()
        if len(words) < 4:
            return False
        
        # Should not be a question
        if claim.strip().endswith('?'):
            return False
        
        # Should not contain special tokens
        if any(token in claim for token in ['<extra_id', '<pad>', '</s>', '<unk>', '<bos>']):
            return False
        
        # Should not be a sentence fragment
        fragment_indicators = [
            claim.lower().startswith('which '),
            claim.lower().startswith('that '),
            claim.lower().startswith('thus '),
            claim.lower().startswith('and '),
            claim.lower().startswith('or '),
            claim.endswith('...'),
            'A word is not' in claim,  # Specific bad pattern
        ]
        
        if any(fragment_indicators):
            return False
        
        # Should contain at least one verb
        verb_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'can', 'will', 
                        'would', 'could', 'should', 'may', 'might', 'must', 'do', 'does', 'did',
                        'provides', 'shows', 'indicates', 'analyzes', 'aims', 'seeks']
        if not any(f' {verb} ' in f' {claim.lower()} ' for verb in verb_indicators):
            return False
        
        # Should have a complete sentence structure (subject + verb + object/complement)
        # Simple heuristic: has multiple words and contains articles/determiners
        has_structure = any(word in claim.lower().split() for word in ['the', 'a', 'an', 'this', 'that', 'these', 'those'])
        if not has_structure and len(words) < 8:
            return False
        
        return True

    def _clean_generated_claim(self, generated: str, original: str) -> str:
        """
        Clean up generated claim text, removing artifacts and ensuring quality.
        """
        # Remove common mT5 artifacts
        claim = generated.strip()
        claim = re.sub(r'<extra_id_\d+>', '', claim)
        claim = re.sub(r'<pad>', '', claim)
        claim = re.sub(r'</s>', '', claim)
        claim = re.sub(r'<unk>', '', claim)
        claim = re.sub(r'<bos>', '', claim)
        claim = claim.strip()
        
        # Remove multiple spaces
        claim = re.sub(r'\s+', ' ', claim)
        
        # If generation is too similar to prompt or too short, use original
        if len(claim) < 15:
            claim = original
        
        # Check if it's just repeating the prompt
        prompt_overlap = similarity(claim.lower(), "rewrite as a verifiable")
        if prompt_overlap > 0.3:
            claim = original
        
        # Ensure proper sentence structure
        if claim and len(claim) > 0:
            if not claim[0].isupper():
                claim = claim[0].upper() + claim[1:]
            
            # Add period if missing
            if claim[-1] not in '.!?':
                claim += '.'
        
        # Final validation - if still bad, return None
        if not self._is_valid_claim(claim):
            return None
        
        return claim

    def _extract_claims_directly(self, text: str, max_claims: int = 5) -> List[str]:
        """
        Fallback method: Extract claims directly from text using rule-based approach.
        Focuses on factual statements with verifiable content.
        Prioritizes temporal, statistical, and specific data.
        """
        claims = []
        
        # Split into sentences
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 20]
        
        # Keywords that indicate factual claims - ENHANCED WITH TEMPORAL
        factual_indicators = [
            'according to', 'research shows', 'study found', 'report', 'data', 
            'statistics', 'percent', '%', 'number', 'increase', 'decrease',
            'announced', 'discovered', 'revealed', 'confirmed', 'stated',
            'aim of', 'purpose of', 'goal is', 'designed to', 'can use',
            'provides', 'allows', 'enables', 'supports',
            # TEMPORAL INDICATORS
            'published', 'released', 'launched', 'introduced', 'founded',
            'established', 'created', 'occurred', 'happened', 'began',
            'in 20', 'in 19',  # Years
            'since 20', 'until 20', 'by 20',  # Year ranges
        ]
        
        # Score each sentence
        scored_sentences = []
        for sentence in sentences:
            # Skip fragments
            if any(sentence.lower().startswith(frag) for frag in ['which ', 'that ', 'thus ', 'and ', 'or ']):
                continue
            
            score = 0.0
            
            # Check for factual indicators
            for indicator in factual_indicators:
                if indicator in sentence.lower():
                    score += 0.3
                    break
            
            # TEMPORAL SCORING
            # Has a year (most important for fact-checking)
            if re.search(r'\b(19|20)\d{2}\b', sentence):
                score += 0.3
                print(f"    [YEAR FOUND] +0.3 in: {sentence[:60]}...")
            
            # Has temporal action verbs
            if re.search(r'\b(published|released|announced|launched|reported|stated|confirmed)\b', sentence, re.IGNORECASE):
                score += 0.2
                print(f"    [TEMPORAL VERB] +0.2")
            
            # Has specific date (month + year)
            if re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+(19|20)\d{2}\b', sentence, re.IGNORECASE):
                score += 0.25
                print(f"    [FULL DATE] +0.25")
            
            # Has percentage/statistics
            if re.search(r'\d+%|\d+\s*percent', sentence, re.IGNORECASE):
                score += 0.2
            
            # Prioritize sentences with other numbers
            if re.search(r'\d+', sentence):
                score += 0.15
            
            # Has proper nouns (capitalized words)
            if re.search(r'\b[A-Z][a-z]+\b', sentence):
                score += 0.1
            
            # Complete sentence structure
            if self._is_valid_claim(sentence + '.'):
                score += 0.2
            
            # Length sweet spot
            if 30 <= len(sentence) <= 200:
                score += 0.2
            
            if score > 0.3:
                scored_sentences.append((sentence, score))
        
        # Sort by score and take top claims
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        for sentence, score in scored_sentences[:max_claims]:
            clean_claim = sentence.strip()
            if not clean_claim.endswith('.'):
                clean_claim += '.'
            
            if clean_claim not in claims:
                claims.append(clean_claim)
                print(f"  ✓ Direct claim (score: {score:.2f}): {clean_claim[:80]}...")
        
        return claims
    def force_ipv4(self):
        def allowed_gai_family():
            return socket.AF_INET
        urllib3_cn.allowed_gai_family = allowed_gai_family


    def factcheck_api_search(self, claim: str, language: str = "en") -> List[Dict[str, Any]]:
        """
        Search Google Fact Check API for fact-checks related to a claim.
        """
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": claim, "languageCode": language, "key": FACTCHECK_API_KEY}
        
        try:
            self.force_ipv4()
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                print(f"Fact Check API returned status {resp.status_code}")
                return []
            
            data = resp.json()
            matches = []
            for c in data.get("claims", []):
                claim_text = c.get("text", "")
                claim_date = c.get("claimDate", None)
                for review in c.get("claimReview", []):
                    matches.append({
                        "claim_text": claim_text,
                        "publisher": (review.get("publisher") or {}).get("name", ""),
                        "rating": review.get("textualRating", ""),
                        "title": review.get("title"),
                        "url": review.get("url"),
                        "date": claim_date
                    })
            return matches
        except Exception as e:
            print(f"Fact Check API error: {e}")
            return []

    def google_cse_search(self, query: str, num: int = 5) -> List[Dict]:
        """
        Search Google Custom Search for relevant articles.
        """
        try:
            es = quote_plus(query)
            url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_CSE_API_KEY}&cx={GOOGLE_CSE_CX}&q={es}&num={num}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            
            items = resp.json().get("items", [])
            results = []
            for it in items:
                snippet = it.get("snippet", "")
                results.append({
                    "title": it.get("title"),
                    "url": it.get("link"),
                    "snippet": snippet,
                    "score": similarity(normalize(query), normalize(snippet))
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        except Exception as e:
            print(f"Google CSE error: {e}")
            return []

    def verify_text(self, text: str, max_claims: int = 5) -> List[Dict[str, Any]]:
        """
        Full fact-checking pipeline: generate claims using mT5, then verify with 
        FactCheck API, fallback to Google Custom Search if needed.
        """
        print("Starting fact-checking pipeline...")
        claims = self.generate_claims(text, max_claims=max_claims)
        results = []

        for claim in claims:
            print(f"Checking claim: {claim}")
            fc_matches = self.factcheck_api_search(claim)
            
            if fc_matches:
                results.append({
                    "claim": claim,
                    "method": "factcheck_api",
                    "fact_matches": fc_matches,
                    "search_matches": self.google_cse_search(claim, num=3)
                })
            else:
                results.append({
                    "claim": claim,
                    "method": "custom_search",
                    "fact_matches": [],
                    "search_matches": self.google_cse_search(claim, num=5)
                })
        
        return results

    def check_probabilities(self, in_text, topk=40, include_detectgpt=True, include_fastdetect=True, 
                          fastdetect_api_key=None, include_factcheck=True, max_claims=5,generate_gltr_viz=True):
        """
        Enhanced check_probabilities that includes GLTR, DetectGPT, FastDetect, and Fact-Checking
        """
        print(f"=== check_probabilities called ===")
        
        # ============ GLTR Analysis ============
        # Tokenize input
        inputs = self.tokenizer(in_text, return_tensors="pt").to(self.device)
        token_ids = inputs["input_ids"][0]
        print("====================tokenized text===================")
        # Forward pass
        with torch.no_grad():
            logits = self.model(token_ids.unsqueeze(0)).logits
        all_probs = torch.softmax(logits[0, :-1, :], dim=-1)
        y = token_ids[1:]
    
        # Real token positions
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        print("=================gltr running======================")
        real_topk_pos = [(sorted_preds[i] == y[i].item()).nonzero(as_tuple=True)[0][0].item() 
                         if len((sorted_preds[i] == y[i].item()).nonzero(as_tuple=True)[0]) > 0 else -1
                         for i in range(y.shape[0])]
        real_topk_probs = all_probs[torch.arange(y.shape[0]), y].detach().to(torch.float32).cpu().numpy().round(5).tolist()
        real_topk = list(zip(real_topk_pos, real_topk_probs))
    
        # Convert tokens to readable strings
        bpe_strings = [self.process_single_token(t) for t in self.tokenizer.convert_ids_to_tokens(token_ids.tolist())]
    
        # Top-k predictions
        topk_vals, topk_inds = torch.topk(all_probs, k=topk, dim=1)
        pred_topk = []
        for i in range(y.shape[0]):
            token_probs = []
            for token_id, prob in zip(topk_inds[i].tolist(), topk_vals[i].detach().to(torch.float32).cpu().numpy()):
                token_probs.append((self.process_single_token(self.tokenizer.convert_ids_to_tokens([token_id])[0]),
                                    float(prob)))
            pred_topk.append(token_probs)

    
        # ============ DetectGPT Score ============
        detectgpt_result = None
        if include_detectgpt:
            print("Computing DetectGPT score...")
            clean_text = preprocess_pdf_text(in_text)
            if len(clean_text) < 10:
                print("⚠️ Cleaned text too short — reverting to original.")
                clean_text = in_text

            print(f"DetectGPT: cleaned text length = {len(clean_text)}")
            detectgpt_result = self.detectgpt_score(clean_text)
    
        # ============ FastDetect Score ============
        fastdetect_result = None
        if include_fastdetect and fastdetect_api_key:
            print("Computing FastDetect score...")
            fastdetect_result = self.fastdetect_score(in_text, fastdetect_api_key)
    
        # ============ Fact-Checking ============
        factcheck_result = None
        if include_factcheck:
            print("Running fact-checking pipeline...")
            try:
                factcheck_result = self.verify_text(in_text, max_claims=max_claims)
            except Exception as e:
                print(f"Fact-checking failed: {e}")
                traceback.print_exc()
                factcheck_result = {"error": str(e)}
    
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        # ============ Compile Results ============
        result = {
            "bpe_strings": bpe_strings,
            "real_topk": real_topk,
            "pred_topk": pred_topk,
            "final_text": "".join(bpe_strings)
        }
        
        # Add DetectGPT results if computed
        if detectgpt_result:
            result["detectgpt"] = detectgpt_result
        
        # Add FastDetect results if computed
        if fastdetect_result:
            result["fastdetect"] = fastdetect_result
        
        # Add Fact-Checking results if computed
        if factcheck_result:
            result["factcheck"] = factcheck_result

        if generate_gltr_viz:
            print("Generating GLTR visualization image...")
            try:
                original_pdf_path = getattr(self, '_current_pdf_path', None)
        
                if original_pdf_path and os.path.exists(original_pdf_path):
                    print(f"Using PDF overlay method with: {original_pdf_path}")
                    gltr_image_path = self.generate_gltr_pdf_screenshot(
                        {"bpe_strings": bpe_strings, "real_topk": real_topk},
                        original_pdf_path
                    )
                else:
                    print("Using standard HTML screenshot method")
                    gltr_image_path = self.generate_gltr_image({
                        "bpe_strings": bpe_strings,
                        "real_topk": real_topk
                    })    
                
                if gltr_image_path:
                    result["gltr_image"] = gltr_image_path
                    print(f"✓ GLTR image ready: {gltr_image_path}")
                else:
                    print("✗ GLTR image generation failed")
                    result["gltr_image"] = None
            except Exception as e:
                print(f"✗ GLTR image error: {e}")
                result["gltr_image"] = None
    
        return result
    def set_current_pdf_path(self, pdf_path: str):
        """Set the current PDF path for GLTR overlay generation"""
        self._current_pdf_path = pdf_path
    # ============================================================================
    # PDF GENERATION WITH GLTR IMAGE
    # ============================================================================
    def generate_pdf_report(self, json_data: dict, output_path: str = None) -> BytesIO:
        """
        Generate comprehensive PDF report from analysis results
        
        Args:
            json_data: Analysis results dictionary
            output_path: Optional file path to save PDF (if None, returns BytesIO buffer)
        
        Returns:
            BytesIO buffer containing PDF
        """
        try:
            # Create buffer
            buffer = BytesIO()
            
            # Determine output destination
            if output_path:
                doc = SimpleDocTemplate(
                    output_path,
                    pagesize=letter,
                    rightMargin=0.75*inch,
                    leftMargin=0.75*inch,
                    topMargin=0.75*inch,
                    bottomMargin=0.75*inch
                )
            else:
                doc = SimpleDocTemplate(
                    buffer,
                    pagesize=letter,
                    rightMargin=0.75*inch,
                    leftMargin=0.75*inch,
                    topMargin=0.75*inch,
                    bottomMargin=0.75*inch
                )
            
            styles = self.create_pdf_styles()
            story = []
            
            # 1. Title and metadata
            story.extend(self.build_pdf_header(styles, json_data))
            
            # 2. Executive summary
            story.extend(self.build_pdf_summary_section(styles, json_data))
            
            # 3. GLTR visualization
            if "bpe_strings" in json_data:
                story.extend(self.build_pdf_gltr_section(styles, json_data))
            
            # 4. Smart page break before detection scores
            story.append(CondPageBreak(4*inch))
            
            # 5. Detection Scores
            if "detectgpt" in json_data or "fastdetect" in json_data:
                story.extend(self.build_pdf_detection_scores(styles, json_data))
            
            # 6. Force new page for fact-checking section
            story.append(PageBreak())
            
            # 7. Fact-checking
            if "factcheck" in json_data:
                pdf_claims_data = {"claims": json_data["factcheck"]}
                story.extend(self.build_pdf_claims_table(styles, pdf_claims_data))
                story.extend(self.build_pdf_factcheck_details(styles, pdf_claims_data))
            
            # 8. Conclusion (already has PageBreak() inside)
            story.extend(self.build_pdf_conclusion(styles, json_data))
            
            # ============ BUILD THE PDF ============
            doc.build(story)
            
            # If using buffer, reset position for reading
            if not output_path:
                buffer.seek(0)
                return buffer
            else:
                print(f"✓ PDF successfully written to: {output_path}")
                # Also return buffer for consistency
                with open(output_path, 'rb') as f:
                    buffer = BytesIO(f.read())
                buffer.seek(0)
                return buffer
                
        except Exception as e:
            print(f"PDF generation failed: {e}")
            traceback.print_exc()
            raise
    def build_pdf_conclusion(self, styles, json_data: dict):
        """Build comprehensive conclusion and recommendations"""
        elements = []
        
        elements.append(PageBreak())  # Start on new page
        
        header = Paragraph("Final Assessment and Recommendations", styles['SectionHeader'])
        elements.append(header)
        
        # Aggregate all signals
        signals = []
        
        # GLTR signal
        real_topk = json_data.get("real_topk", [])
        if real_topk:
            top10_pct = sum(1 for pos, _ in real_topk if pos < 10) / len(real_topk) * 100
            if top10_pct > 70:
                signals.append(("GLTR", "AI", f"{top10_pct:.1f}% top-10 tokens"))
            elif top10_pct < 55:
                signals.append(("GLTR", "Human", f"{top10_pct:.1f}% top-10 tokens"))
            else:
                signals.append(("GLTR", "Unclear", f"{top10_pct:.1f}% top-10 tokens"))
        
        # DetectGPT signal
        dg_score = json_data.get("detectgpt", {}).get("detectgpt_score")
        if dg_score is not None:
            if dg_score < -0.5:
                signals.append(("DetectGPT", "AI", f"Score: {dg_score:.3f}"))
            elif dg_score > 0.1:
                signals.append(("DetectGPT", "Human", f"Score: {dg_score:.3f}"))
            else:
                signals.append(("DetectGPT", "Unclear", f"Score: {dg_score:.3f}"))
        
        # FastDetect signal
        fd_prob = json_data.get("fastdetect", {}).get("prob")
        if fd_prob is not None:
            if fd_prob > 0.75:
                signals.append(("FastDetect", "AI", f"Probability: {fd_prob:.2f}"))
            elif fd_prob < 0.45:
                signals.append(("FastDetect", "Human", f"Probability: {fd_prob:.2f}"))
            else:
                signals.append(("FastDetect", "Unclear", f"Probability: {fd_prob:.2f}"))
        
        # Fact-check signal
        factcheck = json_data.get("factcheck", [])
        if factcheck:
            all_ratings = []
            for claim in factcheck:
                for match in claim.get("fact_matches", []):
                    all_ratings.append(match.get("rating", "").lower())
            
            false_count = sum(1 for r in all_ratings if any(ind in r for ind in ["false", "incorrect"]))
            if false_count > 0:
                signals.append(("Fact-Check", "Issues", f"{false_count} false rating(s)"))
            elif all_ratings:
                signals.append(("Fact-Check", "Clean", "No false ratings"))
        
        # Create signals summary table
        if signals:
            signal_data = [["Detection Method", "Assessment", "Key Metric"]]
            signal_data.extend(signals)
            
            signal_table = Table(signal_data, colWidths=[2*inch, 1.5*inch, 3*inch])
            signal_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('WORDWRAP', (0, 0), (-1, -1)),
            ]))
            
            elements.append(signal_table)
            elements.append(Spacer(1, 0.25*inch))
        
        # Overall verdict
        ai_signals = sum(1 for _, assessment, _ in signals if assessment == "AI")
        human_signals = sum(1 for _, assessment, _ in signals if assessment == "Human")
        
        verdict_header = Paragraph("<b>Overall Verdict:</b>", styles['Subsection'])
        elements.append(verdict_header)
        
        if ai_signals > human_signals and ai_signals >= 2:
            verdict = (
                f"<b>Likely AI-Generated ({ai_signals}/{len(signals)} methods indicate AI)</b><br/><br/>"
                "Multiple independent detection methods converge on AI generation. The preponderance of evidence "
                "suggests this text was produced by a language model. While no detection system is infallible, "
                "the consistency across methodologies provides strong statistical support for this assessment."
            )
            verdict_color = colors.HexColor('#ffcdd2')
            confidence = "High" if ai_signals >= 3 else "Moderate"
        elif human_signals > ai_signals and human_signals >= 2:
            verdict = (
                f"<b>Likely Human-Written ({human_signals}/{len(signals)} methods indicate human)</b><br/><br/>"
                "The balance of evidence suggests human authorship. Detection methods identify patterns consistent "
                "with natural human writing, including appropriate unpredictability and stylistic variation. "
                "While AI-assisted drafting cannot be entirely ruled out, the text does not exhibit characteristic "
                "signatures of pure AI generation."
            )
            verdict_color = colors.HexColor('#c8e6c9')
            confidence = "High" if human_signals >= 3 else "Moderate"
        else:
            verdict = (
                f"<b>Inconclusive ({ai_signals} AI signals, {human_signals} human signals)</b><br/><br/>"
                "Detection methods provide conflicting or insufficient evidence for definitive classification. "
                "This may indicate: hybrid authorship (human + AI collaboration), heavily edited AI content, "
                "edge case characteristics, or limitations of current detection technology. Manual expert review "
                "is strongly recommended before drawing conclusions."
            )
            verdict_color = colors.HexColor('#fff9c4')
            confidence = "Low"
        
        verdict_para = Paragraph(verdict, styles['DetailText'])
        verdict_table = Table([[verdict_para]], colWidths=[6.5*inch])
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), verdict_color),
            ('BOX', (0, 0), (-1, -1), 3, colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('WORDWRAP', (0, 0), (-1, -1)),
        ]))
        
        elements.append(verdict_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Confidence and reliability
        confidence_text = Paragraph(
            f"<b>Confidence Level: {confidence}</b><br/><br/>"
            "Confidence assessment is based on: (1) convergence across detection methods, (2) strength of individual "
            "signals, (3) text length and quality (longer, well-formed texts yield more reliable results), and "
            "(4) absence of confounding factors. High confidence indicates >80% likelihood of correct classification "
            "based on validation studies. Moderate confidence indicates 60-80% likelihood. Low confidence (<60%) "
            "requires additional investigation.",
            styles['DetailText']
        )
        elements.append(confidence_text)
        elements.append(Spacer(1, 0.2*inch))
        
        # Actionable recommendations
        rec_header = Paragraph("Recommendations and Next Steps", styles['Subsection'])
        elements.append(rec_header)
        
        recommendations = []
        
        if ai_signals > human_signals and ai_signals >= 2:
            recommendations = [
                "<b>For Academic/Educational Contexts:</b> If this text was submitted as original student work, "
                "initiate academic integrity procedures per institutional policy. Request the student to: "
                "(1) explain their research and writing process in detail, (2) provide drafts/revision history, "
                "(3) discuss content verbally to demonstrate understanding, (4) submit additional writing samples "
                "for comparison.",
                
                "<b>For Professional/Business Contexts:</b> If authenticity is critical (legal documents, journalism, "
                "original research), verify with the claimed author. Consider requesting: (1) access to source documents "
                "and research materials, (2) explanation of methodology and sourcing, (3) author interview to confirm "
                "subject matter expertise.",
                
                "<b>For Content Moderation:</b> Flag for human review if AI-generated content violates platform policies. "
                "Consider context: disclosed AI use vs. undisclosed AI use, commercial vs. personal content, potential "
                "for misinformation spread.",
                
                "<b>For Research/Analysis:</b> Document detection results with full methodology transparency. Acknowledge "
                "limitations. Do not use as sole evidence in high-stakes decisions. Combine with human expert judgment."
            ]
        elif human_signals > ai_signals and human_signals >= 2:
            recommendations = [
                "<b>For Academic/Educational Contexts:</b> Provisionally accept as authentic student work. No further "
                "action required unless other concerns exist (plagiarism from human sources, collaboration violations, etc.).",
                
                "<b>For Professional Contexts:</b> Proceed with normal verification processes appropriate to content type. "
                "AI detection does not suggest authenticity concerns, though standard fact-checking and editorial review "
                "remain applicable.",
                
                "<b>For Content Moderation:</b> No AI-related policy violations indicated. Standard content policies apply.",
                
                "<b>General Note:</b> While detection results suggest human authorship, this analysis cannot detect: "
                "plagiarism from human sources, unauthorized collaboration, or content generated by unknown/proprietary AI "
                "systems outside our test model's distribution."
            ]
        else:
            recommendations = [
                "<b>Priority: Request Additional Evidence</b> - Given inconclusive results, gather supporting materials: "
                "longer text samples (300+ words), revision history, author interview/discussion, supplementary writing "
                "samples from same author for baseline comparison.",
                
                "<b>Consider Context:</b> Evaluate the specific use case. For low-stakes content (informal writing, "
                "brainstorming, draft ideas), inconclusive results may be acceptable. For high-stakes contexts (academic "
                "assessment, legal documents, published journalism), pursue additional verification before conclusions.",
                
                "<b>Manual Expert Review:</b> Engage human experts with relevant domain knowledge to: (1) assess content "
                "sophistication and accuracy, (2) evaluate stylistic consistency, (3) identify subject matter expertise "
                "indicators, (4) look for contextual clues (personal anecdotes, specific references, discipline-specific "
                "knowledge) that suggest human authorship.",
                
                "<b>Re-analysis Options:</b> If additional text becomes available, re-run analysis with expanded sample. "
                "Detection accuracy improves substantially with longer texts (>500 words vs. <200 words).",
                
                "<b>Avoid Premature Conclusions:</b> Do not penalize or reward based on inconclusive results. When methods "
                "disagree, the prudent approach is to treat as 'unknown' rather than defaulting to either classification."
            ]
        
        for rec in recommendations:
            rec_para = Paragraph(f"• {rec}", styles['DetailText'])
            elements.append(rec_para)
            elements.append(Spacer(1, 0.12*inch))
        
        elements.append(Spacer(1, 0.25*inch))
        
        # Important disclaimers
        disclaimer_header = Paragraph("Important Disclaimers", styles['Subsection'])
        elements.append(disclaimer_header)
        
        disclaimer_text = Paragraph(
            "<b>Legal and Ethical Considerations:</b> This report provides <i>probabilistic technical analysis</i>, "
            "not definitive proof of authorship. AI detection tools can produce false positives and false negatives. "
            "Known error rates: ~5-15% false positive rate (human text misclassified as AI), ~10-25% false negative rate "
            "(AI text misclassified as human), varying by text length, language, domain, and model used for generation.<br/><br/>"
            
            "<b>Not Suitable As Sole Evidence For:</b> Academic misconduct determinations, employment decisions, legal "
            "proceedings, or any action with significant consequences for individuals. Always combine with human judgment, "
            "contextual evidence, and due process.<br/><br/>"
            
            "<b>Detection Limitations:</b> Methods may fail to detect: (1) heavily edited AI content, (2) AI content "
            "mixed with human writing, (3) content from AI models using advanced decoding strategies (high temperature, "
            "diverse sampling), (4) content from proprietary models with different probability distributions, "
            "(5) AI-assisted writing where human provides substantial creative input.<br/><br/>"
            
            "<b>Bias Considerations:</b> Detection accuracy may vary across: (1) Languages (optimized for English), "
            "(2) Writing styles (formal vs. casual), (3) Domains (technical vs. creative), (4) Author demographics "
            "(non-native speakers may trigger false positives). Consider these factors in interpretation.<br/><br/>"
            
            "<b>Evolving Technology:</b> Both AI generation and detection methods evolve rapidly. This report reflects "
            "current state-of-the-art as of the report date. Detection methods may become less effective against "
            "future generation models. Regular methodology updates recommended for ongoing use.",
            styles['DetailText']
        )
        
        disclaimer_table = Table([[disclaimer_text]], colWidths=[6.5*inch])
        disclaimer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('BOX', (0, 0), (-1, -1), 2, colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('WORDWRAP', (0, 0), (-1, -1)),
        ]))
        
        elements.append(disclaimer_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # References and further reading
        references_header = Paragraph("Technical References", styles['Subsection'])
        elements.append(references_header)
        
        references = [
            "Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical Detection and Visualization of "
            "Generated Text. <i>Proceedings of ACL 2019</i>.",
            
            "Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-Shot Machine-Generated "
            "Text Detection using Probability Curvature. <i>Proceedings of ICML 2023</i>.",
            
            "Bao, G., Zhao, Y., Teng, Z., Yang, L., & Zhang, Y. (2023). Fast-DetectGPT: Efficient Zero-Shot Detection of "
            "Machine-Generated Text via Conditional Probability Curvature. <i>arXiv preprint arXiv:2310.05130</i>.",
            
            "Solaiman, I., et al. (2019). Release Strategies and the Social Impacts of Language Models. "
            "<i>arXiv preprint arXiv:1908.09203</i>.",
            
            "Ippolito, D., Duckworth, D., Callison-Burch, C., & Eck, D. (2020). Automatic Detection of Generated Text "
            "is Easiest when Humans are Fooled. <i>Proceedings of ACL 2020</i>."
        ]
        
        for ref in references:
            ref_para = Paragraph(f"• {ref}", styles['DetailText'])
            elements.append(ref_para)
            elements.append(Spacer(1, 0.08*inch))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Footer with contact/support
        footer_text = Paragraph(
            "<b>Report Information:</b> This report was generated by ScanIt automated analysis system. "
            "For questions about methodology, interpretation, or to request re-analysis with different parameters, "
            "consult documentation or contact your system administrator. Report version: 2.0<br/><br/>"
            
            "<i>Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " | "
            "Analyzer: " + self.base_model_name + "</i>",
            styles['DetailText']
        )
        
        footer_table = Table([[footer_text]], colWidths=[6.5*inch])
        footer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e8eaf6')),
            ('BOX', (0, 0), (-1, -1), 1, colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('WORDWRAP', (0, 0), (-1, -1)),
        ]))
        
        elements.append(footer_table)
        
        return elements

    def create_pdf_styles(self):
        """Create custom styles for PDF report"""
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=26,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            keepWithNext=True
        ))
        
        # Section header style
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#283593'),
            spaceAfter=15,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection style
        styles.add(ParagraphStyle(
            name='Subsection',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#3949ab'),
            spaceAfter=10,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            keepWithNext=True
        ))
        
        # Detailed explanation style
        styles.add(ParagraphStyle(
            name='DetailText',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#424242'),
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            leading=14
        ))
        
        # Interpretation box style
        styles.add(ParagraphStyle(
            name='InterpretationBox',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#1565c0'),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=12,
            spaceBefore=12,
            leading=14,
            alignment=TA_JUSTIFY
        ))
        
        return styles

    def build_pdf_header(self, styles, json_data: dict):
        """Build comprehensive PDF header section"""
        elements = []
        
        # Title
        title = Paragraph("ScanIt - Comprehensive AI Text Analysis Report", styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Report description
        intro = Paragraph(
            "This report provides a multi-dimensional forensic analysis of the submitted text using "
            "state-of-the-art AI detection methodologies. The analysis combines statistical language modeling, "
            "perturbation-based detection, and factual verification to provide a comprehensive assessment.",
            styles['DetailText']
        )
        elements.append(intro)
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = [
            ["Report Generated:", timestamp],
            ["Analysis Framework:", "Multi-Method AI Detection"],
            ["Primary Model:", self.base_model_name],
            ["Mask-Filling Model:", self.mask_filling_model_name],
            ["Detection Methods:", "GLTR, DetectGPT, FastDetectGPT, Fact-Checking"]
        ]
        
        meta_table = Table(metadata, colWidths=[2.2*inch, 4.3*inch])
        meta_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('BOX', (0, 0), (-1, -1), 1, colors.grey),
            ('WORDWRAP', (0, 0), (-1, -1)),
        ]))
        
        elements.append(meta_table)
        elements.append(Spacer(1, 0.3*inch))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#283593')))
        elements.append(Spacer(1, 0.4*inch))
        
        return elements

    def build_pdf_summary_section(self, styles, json_data: dict):
        """Build detailed executive summary section"""
        elements = []
        
        header = Paragraph("Executive Summary", styles['SectionHeader'])
        elements.append(header)
        
        # Introduction to token analysis
        intro = Paragraph(
            "<b>Overview:</b> This section provides a statistical overview of token predictability in the analyzed text. "
            "Token analysis reveals how predictable each word choice is based on the language model's probability distribution. "
            "This forms the foundation for understanding whether text exhibits patterns typical of human or AI authorship.",
            styles['DetailText']
        )
        elements.append(intro)
        elements.append(Spacer(1, 0.15*inch))
        
        # Calculate statistics
        real_topk = json_data.get("real_topk", [])
        total_tokens = len(real_topk)
        
        if total_tokens > 0:
            top10_count = sum(1 for pos, _ in real_topk if pos < 10)
            top100_count = sum(1 for pos, _ in real_topk if 10 <= pos < 100)
            top1000_count = sum(1 for pos, _ in real_topk if 100 <= pos < 1000)
            beyond1000_count = sum(1 for pos, _ in real_topk if pos >= 1000)
            
            top10_pct = top10_count/total_tokens*100
            top100_pct = top100_count/total_tokens*100
            top1000_pct = top1000_count/total_tokens*100
            beyond1000_pct = beyond1000_count/total_tokens*100
            
            summary_data = [
                ["Metric", "Count", "Percentage", "Interpretation"],
                ["Total Tokens Analyzed", str(total_tokens), "100%", "Complete text sample"],
                ["Top-10 Predictions", str(top10_count), f"{top10_pct:.1f}%", 
                "Highly predictable choices"],
                ["Top-100 Predictions", str(top100_count), f"{top100_pct:.1f}%", 
                "Moderately predictable"],
                ["Top-1000 Predictions", str(top1000_count), f"{top1000_pct:.1f}%", 
                "Less common choices"],
                ["Beyond Top-1000", str(beyond1000_count), f"{beyond1000_pct:.1f}%", 
                "Unusual/creative choices"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 1*inch, 1.2*inch, 2.3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (2, -1), 'LEFT'),
                ('ALIGN', (3, 0), (3, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('WORDWRAP', (0, 0), (-1, -1)),
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 0.2*inch))
            
            # Detailed interpretation
            interp_header = Paragraph("<b>Statistical Interpretation:</b>", styles['Subsection'])
            elements.append(interp_header)
            
            # AI vs Human likelihood assessment
            if top10_pct > 70:
                ai_assessment = (
                    f"<b>High AI Likelihood:</b> The text contains {top10_pct:.1f}% tokens in the top-10 predictions, "
                    f"which significantly exceeds typical human writing patterns (usually 40-60%). This high concentration "
                    f"of predictable word choices is characteristic of language models, which tend to favor high-probability "
                    f"tokens during generation."
                )
                color_box = colors.HexColor('#ffebee')
            elif top10_pct > 60:
                ai_assessment = (
                    f"<b>Moderate AI Likelihood:</b> With {top10_pct:.1f}% top-10 tokens, the text shows elevated "
                    f"predictability compared to average human writing. This could indicate AI generation, heavy editing, "
                    f"or formulaic human writing in technical/formal contexts."
                )
                color_box = colors.HexColor('#fff3e0')
            else:
                ai_assessment = (
                    f"<b>Human-Like Pattern:</b> The text contains {top10_pct:.1f}% top-10 tokens, which falls within "
                    f"typical human writing ranges. The presence of {beyond1000_pct:.1f}% unusual tokens (beyond top-1000) "
                    f"suggests creative word choices and stylistic variation common in human authorship."
                )
                color_box = colors.HexColor('#e8f5e9')
            
            assessment_para = Paragraph(ai_assessment, styles['DetailText'])
            
            # Create colored box for assessment
            assessment_table = Table([[assessment_para]], colWidths=[6.5*inch])
            assessment_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), color_box),
                ('BOX', (0, 0), (-1, -1), 2, colors.grey),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('WORDWRAP', (0, 0), (-1, -1)),
            ]))
            
            elements.append(assessment_table)
            elements.append(Spacer(1, 0.15*inch))
            
            # Technical explanation
            technical_exp = Paragraph(
                "<b>Technical Background:</b> Language models assign probability distributions to predict the next token "
                "in a sequence. Human writers naturally select from a broader probability distribution, including less "
                "predictable word choices for style, emphasis, or creativity. AI models, particularly when using greedy "
                "or low-temperature sampling, concentrate selections in the high-probability region, creating statistically "
                "detectable patterns.",
                styles['DetailText']
            )
            elements.append(technical_exp)
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    

    def build_pdf_gltr_section(self, styles, json_data: dict):
        """Build comprehensive GLTR visualization section with proper page handling"""
        elements = []
        
        # Main header
        header = Paragraph("GLTR (Giant Language model Test Room) Analysis", styles['SectionHeader'])
        methodology = Paragraph(
            "<b>Methodology:</b> GLTR is a forensic tool developed by MIT-IBM Watson AI Lab and Harvard NLP that visualizes "
            "the probability distribution of each token in text. By coloring tokens according to their ranking in the model's "
            "prediction distribution, GLTR reveals patterns invisible to human readers. This technique exploits the fundamental "
            "difference between how humans and AI systems select words: humans draw from diverse vocabulary with varied "
            "predictability, while AI models systematically favor high-probability choices.",
            styles['DetailText']
        )
        elements.append(KeepTogether([header, methodology]))
        elements.append(Spacer(1, 0.15 * inch))
        
        # Color coding explanation
        color_exp = Paragraph(
            "<b>Color Coding System:</b> Each token is analyzed against the model's probability distribution and assigned "
            "a color based on its ranking position. This creates a visual fingerprint of the text's statistical properties.",
            styles['DetailText']
        )
        elements.append(color_exp)
        elements.append(Spacer(1, 0.12 * inch))
        normal = styles['DetailText']

        # Detailed legend
        legend_data = [
            ["Color", "Rank Range", "Probability", "Typical Occurrence", "Interpretation"],
            ["Green", "1-10", "Very High\n(>5%)",
            Paragraph("40-60% in human text<br/>70-90% in AI text", normal),
            Paragraph("Highly predictable tokens. Overrepresentation suggests AI generation.", normal)],
            ["Yellow", "11-100", "High\n(0.5-5%)",
            Paragraph("20-30% in human text<br/>10-20% in AI text", normal),
            Paragraph("Common but not obvious choices. Moderate predictability.", normal)],
            ["Red", "101-1000", "Medium\n(0.05-0.5%)",
            Paragraph("15-25% in human text<br/>5-10% in AI text", normal),
            Paragraph("Less common word choices requiring contextual sophistication.", normal)],
            ["Purple", ">1000", "Low\n(<0.05%)",
            Paragraph("10-20% in human text<br/>1-5% in AI text", normal),
            Paragraph("Rare or creative choices. High occurrence indicates human authorship.", normal)]
        ]
        
        legend_table = Table(legend_data, colWidths=[0.7*inch, 0.8*inch, 0.9*inch, 1.6*inch, 2.5*inch])
        legend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('WORDWRAP', (0, 0), (-1, -1)),
        ]))
        
        elements.append(legend_table)
        
        # === PAGE BREAK BEFORE IMAGE ===
        elements.append(PageBreak())
        
        # === GLTR IMAGE SECTION ===
        gltr_image_path = json_data.get("gltr_image")

        if gltr_image_path and os.path.exists(gltr_image_path):
            try:
                pil_img = PILImage.open(gltr_image_path)
                
                # Page dimensions - very conservative to ensure full pages fit
                page_width_inch = 6.8   # Slightly narrower for safety
                max_page_height_inch = 8.7   # Maximum height per page (conservative)
                
                img_width_px = pil_img.width
                img_height_px = pil_img.height
                
                # Calculate scaled dimensions maintaining aspect ratio
                scale_factor = page_width_inch / (img_width_px / 96.0)
                scaled_height_inch = (img_height_px / 96.0) * scale_factor
                
                print(f"GLTR Image: {img_width_px}x{img_height_px}px")
                print(f"Scaled dimensions: {page_width_inch:.2f}\" × {scaled_height_inch:.2f}\"")
                
                # If image fits on one page
                if scaled_height_inch <= max_page_height_inch:
                    print("✓ Image fits on single page")
                    img = Image(gltr_image_path, 
                            width=page_width_inch * inch, 
                            height=scaled_height_inch * inch)
                    img.hAlign = 'CENTER'
                    elements.append(Spacer(1, 0.2 * inch))
                    elements.append(img)
                    elements.append(Spacer(1, 0.3 * inch))
                    
                else:
                    # Image too tall - split into full-page chunks
                    print(f"⚠ Image too tall ({scaled_height_inch:.2f}\"), splitting...")
                    
                    # Calculate how many pixels fit per page
                    pixels_per_page = int(max_page_height_inch / scaled_height_inch * img_height_px)
                    num_pages = -(-img_height_px // pixels_per_page)  # Ceiling division
                    
                    print(f"Splitting into {num_pages} pages ({pixels_per_page}px per page)")
                    
                    for page_num in range(num_pages):
                        # Calculate crop boundaries
                        top_px = page_num * pixels_per_page
                        bottom_px = min((page_num + 1) * pixels_per_page, img_height_px)
                        
                        # Crop this page's slice
                        cropped = pil_img.crop((0, top_px, img_width_px, bottom_px))
                        temp_path = f"{gltr_image_path}_part{page_num + 1}.png"
                        cropped.save(temp_path)
                        
                        # Calculate actual height of this slice
                        slice_height_px = bottom_px - top_px
                        slice_height_inch = (slice_height_px / 96.0) * scale_factor
                        
                        print(f"  Page {page_num + 1}: {top_px}-{bottom_px}px → {slice_height_inch:.2f}\"")
                        
                        # Add small spacer at top
                        elements.append(Spacer(1, 0.2 * inch))
                        
                        # Add the image slice (NO HEADER)
                        img_part = Image(temp_path, 
                                    width=page_width_inch * inch,
                                    height=slice_height_inch * inch)
                        img_part.hAlign = 'CENTER'
                        elements.append(img_part)
                        
                        # Add page break between slices (not after last)
                        if page_num < num_pages - 1:
                            elements.append(PageBreak())
                        else:
                            elements.append(Spacer(1, 0.3 * inch))

            except Exception as e:
                print(f"❌ Error embedding GLTR image: {e}")
                import traceback
                traceback.print_exc()
                
                fallback = Paragraph(
                    "⚠ GLTR visualization image could not be embedded. See file separately.",
                    styles['Normal']
                )
                elements.append(fallback)
        else:
            no_img = Paragraph(
                "⚠ GLTR visualization not available for this analysis.",
                styles['Normal']
            )
            elements.append(no_img)


        # === INTERPRETATION SECTION ===
        interpretation_points = [
            "<b>Uniform Green Sequences:</b> Long stretches of green tokens indicate highly predictable, formulaic writing "
            "typical of AI generation. Human writers naturally introduce variety even in straightforward content.",
            
            "<b>Color Mixing Patterns:</b> Healthy mixing of all four colors suggests human authorship, reflecting natural "
            "variation in word choice predictability. AI text typically shows color clustering with green dominance.",
            
            "<b>Purple Token Distribution:</b> Scattered purple tokens indicate creative or uncommon word choices. Their "
            "presence correlates strongly with human authorship, as AI models are trained to avoid low-probability selections.",
            
            "<b>Context Matters:</b> Technical documentation naturally contains more green tokens due to standardized "
            "terminology. Creative writing or opinion pieces should show greater color diversity."
        ]

        interpret_header = Paragraph("<b>How to Interpret the Visualization:</b>", styles['Subsection'])
        elements.append(interpret_header)
        elements.append(Spacer(1, 0.1 * inch))

        for point in interpretation_points:
            bullet = Paragraph(f"• {point}", styles['DetailText'])
            elements.append(bullet)
            elements.append(Spacer(1, 0.08 * inch))

        elements.append(Spacer(1, 0.25 * inch))

        # === RESEARCH REFERENCE ===
        research = Paragraph(
            "<b>Research Foundation:</b> The GLTR methodology is based on peer-reviewed research: Gehrmann, S., Strobelt, H., "
            "&amp; Rush, A. M. (2019). 'GLTR: Statistical Detection and Visualization of Generated Text.' ACL 2019. The approach "
            "has been validated across multiple language models and demonstrates consistent detection capabilities with "
            "accuracy rates of 72–95% depending on text length and generation parameters.",
            styles['DetailText']
        )
        elements.append(research)
        elements.append(Spacer(1, 0.3 * inch))

        return elements
    
    def build_pdf_detection_scores(self, styles, json_data: dict):
        """Build comprehensive detection scores section with proper page break control"""
        elements = []
        
        # Main section header - keep with intro
        header = Paragraph("Advanced AI Detection Scores", styles['SectionHeader'])
        intro = Paragraph(
            "This section presents results from perturbation-based detection methods that analyze how the text responds "
            "to controlled modifications. These techniques exploit the fact that AI-generated text exhibits different "
            "statistical properties under perturbation compared to human-written text.",
            styles['DetailText']
        )
        
        elements.append(KeepTogether([header, intro]))
        elements.append(Spacer(1, 0.2*inch))
        
        # ============ DetectGPT Section ============
        if "detectgpt" in json_data:
            detectgpt_elements = []  # Collect all DetectGPT content
            
            detectgpt_header = Paragraph("DetectGPT Analysis", styles['Subsection'])
            detectgpt_elements.append(detectgpt_header)
            
            dg = json_data["detectgpt"]
            
            # Methodology explanation
            detectgpt_method = Paragraph(
                "<b>Methodology:</b> DetectGPT uses a zero-shot detection approach based on the curvature of the model's "
                "log-probability function. The algorithm generates multiple perturbations of the input text by randomly "
                "masking and re-filling spans, then compares the likelihood of the original text to the average likelihood "
                "of perturbations. AI-generated text typically has negative curvature (original likelihood is lower than "
                "perturbed versions), while human text shows positive curvature.",
                styles['DetailText']
            )
            detectgpt_elements.append(detectgpt_method)
            detectgpt_elements.append(Spacer(1, 0.12*inch))
            
            # Scores table
            score = dg.get("detectgpt_score", "N/A")
            orig_ll = dg.get("original_ll", "N/A")
            pert_ll = dg.get("perturbed_ll_mean", "N/A")
            n_pert = dg.get("n_perturbations", "N/A")
            
            if isinstance(score, (int, float)):
                score_data = [
                    ["Metric", "Value", "Technical Meaning"],
                    ["DetectGPT Score", f"{score:.4f}", "Original LL - Perturbed LL Mean"],
                    ["Original Log-Likelihood", f"{orig_ll:.4f}", "Model confidence in original text"],
                    ["Perturbed LL Mean", f"{pert_ll:.4f}", "Average confidence in modified versions"],
                    ["Perturbations Generated", str(n_pert), "Number of alternative versions analyzed"]
                ]
                
                score_table = Table(score_data, colWidths=[2.2*inch, 1.5*inch, 2.8*inch])
                score_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('WORDWRAP', (0, 0), (-1, -1)),
                ]))
                
                detectgpt_elements.append(score_table)
                detectgpt_elements.append(Spacer(1, 0.15*inch))
                
                # Detailed interpretation
                if score < -0.5:
                    interpretation = (
                        f"<b>Strong AI Signature (Score: {score:.4f}):</b> The substantially negative DetectGPT score "
                        f"indicates the original text has significantly lower likelihood than its perturbations. This pattern "
                        f"is highly characteristic of AI generation, where models produce locally optimal token sequences. "
                        f"When these sequences are perturbed, the model often assigns higher probability to the modifications, "
                        f"revealing the original was already at a local maximum in probability space."
                    )
                    color_box = colors.HexColor('#ffebee')
                elif score < 0:
                    interpretation = (
                        f"<b>Moderate AI Indication (Score: {score:.4f}):</b> The negative score suggests possible AI generation, "
                        f"though the magnitude is modest. This could indicate: (1) AI text with some human editing, "
                        f"(2) human text with formulaic structure, or (3) mixed authorship. Additional analysis methods should "
                        f"be weighted more heavily for conclusive assessment."
                    )
                    color_box = colors.HexColor('#fff3e0')
                else:
                    interpretation = (
                        f"<b>Human-Consistent Pattern (Score: {score:.4f}):</b> The positive or near-zero score indicates "
                        f"the original text is at least as likely as its perturbations, consistent with human authorship. "
                        f"Human writers produce text where random modifications typically decrease coherence and likelihood, "
                        f"resulting in positive DetectGPT scores."
                    )
                    color_box = colors.HexColor('#e8f5e9')
                
                interp_para = Paragraph(interpretation, styles['DetailText'])
                interp_table = Table([[interp_para]], colWidths=[6.5*inch])
                interp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), color_box),
                    ('BOX', (0, 0), (-1, -1), 2, colors.grey),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('WORDWRAP', (0, 0), (-1, -1)),
                ]))
                
                detectgpt_elements.append(interp_table)
                detectgpt_elements.append(Spacer(1, 0.15*inch))
                
                # Technical details
                technical = Paragraph(
                    "<b>Technical Details:</b> DetectGPT's effectiveness stems from exploiting the 'inverse scaling' phenomenon: "
                    "as language models improve, they become more detectable via this method. The algorithm uses T5 for mask-filling "
                    "perturbations, generating alternative phrasings while preserving semantic meaning. Statistical significance "
                    "increases with text length; results are most reliable for texts exceeding 100 tokens.",
                    styles['DetailText']
                )
                detectgpt_elements.append(technical)
            
            # Wrap entire DetectGPT section together
            elements.append(KeepTogether(detectgpt_elements))
            elements.append(Spacer(1, 0.25*inch))
        
        # ============ FastDetect Section ============
        if "fastdetect" in json_data:
            fastdetect_elements = []  # Collect all FastDetect content
            
            fastdetect_header = Paragraph("FastDetectGPT Analysis", styles['Subsection'])
            fastdetect_elements.append(fastdetect_header)
            
            fd = json_data["fastdetect"]
            
            if fd.get("success"):
                # Methodology
                fastdetect_method = Paragraph(
                    "<b>Methodology:</b> FastDetectGPT provides rapid AI detection using conditional probability curvature "
                    "estimation. Unlike DetectGPT which requires multiple perturbations, FastDetect computes detection scores "
                    "using efficient sampling strategies, reducing computational overhead by up to 340x while maintaining "
                    "comparable accuracy. The method is particularly effective for real-time applications.",
                    styles['DetailText']
                )
                fastdetect_elements.append(fastdetect_method)
                fastdetect_elements.append(Spacer(1, 0.12*inch))
                
                prob = fd.get("prob", "N/A")
                
                if isinstance(prob, (int, float)):
                    fd_data = [
                        ["Metric", "Value", "Interpretation"],
                        ["AI Probability", f"{prob:.4f} ({prob*100:.1f}%)", "Likelihood of AI generation"],
                        ["Human Probability", f"{1-prob:.4f} ({(1-prob)*100:.1f}%)", "Likelihood of human authorship"],
                        ["Confidence Level", "High" if abs(prob-0.5) > 0.3 else "Moderate", "Based on distance from threshold"]
                    ]
                    
                    fd_table = Table(fd_data, colWidths=[2*inch, 2*inch, 2.5*inch])
                    fd_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('WORDWRAP', (0, 0), (-1, -1)),
                    ]))
                    
                    fastdetect_elements.append(fd_table)
                    fastdetect_elements.append(Spacer(1, 0.15*inch))
                    
                    # Interpretation
                    if prob > 0.75:
                        fd_interp = (
                            f"<b>High AI Confidence ({prob*100:.1f}%):</b> FastDetectGPT assigns {prob*100:.1f}% probability "
                            f"to AI generation, indicating strong statistical evidence. This confidence level suggests the text "
                            f"exhibits multiple characteristic patterns of machine generation including consistent token probability "
                            f"distributions and systematic linguistic choices typical of transformer-based models."
                        )
                        fd_color = colors.HexColor('#ffebee')
                    elif prob > 0.55:
                        fd_interp = (
                            f"<b>Moderate AI Indication ({prob*100:.1f}%):</b> The probability slightly favors AI generation "
                            f"but remains in the uncertain range. This could indicate: (1) heavily edited AI text, "
                            f"(2) AI-assisted human writing, (3) human text with standardized structure, or (4) text at the "
                            f"boundary of model training distribution. Cross-reference with other detection methods for clarity."
                        )
                        fd_color = colors.HexColor('#fff3e0')
                    else:
                        fd_interp = (
                            f"<b>Human-Likely Pattern ({prob*100:.1f}% AI probability):</b> FastDetectGPT assigns low probability "
                            f"to AI generation, with {(1-prob)*100:.1f}% confidence in human authorship. The text demonstrates "
                            f"statistical properties inconsistent with typical language model outputs, including appropriate "
                            f"unpredictability and natural linguistic variation."
                        )
                        fd_color = colors.HexColor('#e8f5e9')
                    
                    fd_interp_para = Paragraph(fd_interp, styles['DetailText'])
                    fd_interp_table = Table([[fd_interp_para]], colWidths=[6.5*inch])
                    fd_interp_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), fd_color),
                        ('BOX', (0, 0), (-1, -1), 2, colors.grey),
                        ('LEFTPADDING', (0, 0), (-1, -1), 15),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                        ('TOPPADDING', (0, 0), (-1, -1), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                        ('WORDWRAP', (0, 0), (-1, -1)),
                    ]))
                    
                    fastdetect_elements.append(fd_interp_table)
            else:
                error_msg = Paragraph(
                    f"<b>Warning:</b> FastDetectGPT analysis failed: {fd.get('error', 'Unknown error')}",
                    styles['Normal']
                )
                fastdetect_elements.append(error_msg)
            
            # Wrap entire FastDetect section together
            elements.append(KeepTogether(fastdetect_elements))
            elements.append(Spacer(1, 0.25*inch))
        
        # ============ Cross-Method Synthesis ============
        if "detectgpt" in json_data and "fastdetect" in json_data:
            synthesis_elements = []
            
            synthesis_header = Paragraph("Cross-Method Synthesis", styles['Subsection'])
            synthesis_elements.append(synthesis_header)
            
            # Analyze agreement between methods
            dg_score = json_data.get("detectgpt", {}).get("detectgpt_score")
            fd_prob = json_data.get("fastdetect", {}).get("prob")
            
            if dg_score is not None and fd_prob is not None:
                # Determine agreement
                dg_suggests_ai = dg_score < 0
                fd_suggests_ai = fd_prob > 0.5
                
                if dg_suggests_ai == fd_suggests_ai:
                    agreement = "high"
                    agreement_text = "Both detection methods agree"
                else:
                    agreement = "conflicting"
                    agreement_text = "Detection methods show conflicting results"
                
                synthesis_intro = Paragraph(
                    f"<b>{agreement_text}.</b> Comparing results across multiple methodologies provides stronger evidence "
                    f"than any single method. Agreement increases confidence, while disagreement suggests edge cases requiring "
                    f"human expert review.",
                    styles['DetailText']
                )
                synthesis_elements.append(synthesis_intro)
                synthesis_elements.append(Spacer(1, 0.12*inch))
                
                if agreement == "high":
                    if dg_suggests_ai:
                        consensus = (
                            "<b>Consensus: AI Generation Likely</b><br/><br/>"
                            f"Both DetectGPT (score: {dg_score:.4f}) and FastDetect (probability: {fd_prob:.2f}) indicate "
                            "AI generation. This convergence significantly strengthens the assessment. When multiple independent "
                            "detection approaches agree, the likelihood of accurate classification exceeds 85-90% in validation studies. "
                            "<br/><br/>"
                            "<b>Recommended Action:</b> Treat as AI-generated pending additional verification. If human authorship "
                            "is claimed, request supporting documentation or conduct follow-up analysis with extended samples."
                        )
                        consensus_color = colors.HexColor('#ffebee')
                    else:
                        consensus = (
                            "<b>Consensus: Human Authorship Likely</b><br/><br/>"
                            f"Both DetectGPT (score: {dg_score:.4f}) and FastDetect (probability: {fd_prob:.2f}) suggest "
                            "human authorship. This agreement provides strong evidence against AI generation. The text demonstrates "
                            "statistical properties consistent with human writing across multiple analytical dimensions."
                            "<br/><br/>"
                            "<b>Recommended Action:</b> Provisionally accept as human-written. Remaining uncertainty is primarily "
                            "due to inherent limitations in computational detection, not specific red flags in this text."
                        )
                        consensus_color = colors.HexColor('#e8f5e9')
                else:
                    consensus = (
                        "<b>Conflicting Signals Detected</b><br/><br/>"
                        f"DetectGPT (score: {dg_score:.4f}) and FastDetect (probability: {fd_prob:.2f}) provide contradictory "
                        "assessments. This disagreement is significant and suggests: <br/>"
                        "• <b>Hybrid authorship:</b> Mix of human and AI content<br/>"
                        "• <b>Heavy editing:</b> AI draft extensively revised by human, or vice versa<br/>"
                        "• <b>Edge case:</b> Text characteristics at boundary of training distribution<br/>"
                        "• <b>Domain specificity:</b> Specialized content requiring additional context<br/><br/>"
                        "<b>Recommended Action:</b> Manual expert review required. Reference GLTR visualization and fact-checking "
                        "results for additional evidence. Consider requesting longer text samples for re-analysis."
                    )
                    consensus_color = colors.HexColor('#fff9c4')
                
                consensus_para = Paragraph(consensus, styles['DetailText'])
                consensus_table = Table([[consensus_para]], colWidths=[6.5*inch])
                consensus_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), consensus_color),
                    ('BOX', (0, 0), (-1, -1), 2, colors.grey),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('WORDWRAP', (0, 0), (-1, -1)),
                ]))
                
                synthesis_elements.append(consensus_table)
            
            # Wrap synthesis section
            elements.append(KeepTogether(synthesis_elements))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # ============ Limitations ============
        limitations_elements = []
        
        limitations_header = Paragraph("Limitations and Considerations", styles['Subsection'])
        limitations_elements.append(limitations_header)
        
        limitations_text = Paragraph(
            "<b>Important Caveats:</b> All AI detection methods have inherent limitations. False positives can occur with "
            "highly formulaic human writing (legal documents, technical specifications, standardized reports). False negatives "
            "can occur with AI text that has been extensively edited, paraphrased, or generated with high-temperature/diverse "
            "sampling. Detection accuracy decreases for: (1) short texts (<100 tokens), (2) non-English content, "
            "(3) domain-specific technical writing, (4) texts mixing multiple authorship sources. These tools provide "
            "<i>probabilistic evidence</i>, not definitive proof. Decisions with significant consequences (academic integrity, "
            "legal proceedings, hiring) should involve human expert judgment alongside automated analysis.",
            styles['DetailText']
        )
        limitations_elements.append(limitations_text)
        
        elements.append(KeepTogether(limitations_elements))
        elements.append(Spacer(1, 0.3*inch))
        
        return elements

    def build_pdf_claims_table(self, styles, pdf_claims_data: dict):
        """Build comprehensive claims summary with educational context"""
        elements = []
        
        header = Paragraph("Fact-Checking Analysis", styles['SectionHeader'])
        elements.append(header)
        
        intro = Paragraph(
            "<b>Purpose:</b> This section analyzes factual claims extracted from the text to assess veracity and detect "
            "potential misinformation. AI-generated text sometimes contains plausible-sounding but factually incorrect "
            "statements ('hallucinations'). This analysis cross-references claims against verified sources to identify "
            "inaccuracies, outdated information, or fabricated details.",
            styles['DetailText']
        )
        elements.append(intro)
        elements.append(Spacer(1, 0.15*inch))
        
        claims = pdf_claims_data.get("claims", [])
        
        if not claims:
            no_claims = Paragraph(
                "⚠ <b>No verifiable claims detected.</b> The text may be: (1) too short for claim extraction, "
                "(2) primarily opinion/subjective content, (3) lacking specific factual statements, or "
                "(4) failed claim generation process. Fact-checking is most effective for texts containing "
                "specific dates, statistics, named entities, or verifiable events.",
                styles['DetailText']
            )
            
            no_claims_table = Table([[no_claims]], colWidths=[6.5*inch])
            no_claims_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff9c4')),
                ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('WORDWRAP', (0, 0), (-1, -1)),
            ]))
            
            elements.append(no_claims_table)
            return elements
        
        # Claims overview
        overview = Paragraph(
            f"<b>Claims Identified:</b> {len(claims)} factual claims were extracted and analyzed. Each claim "
            f"was verified against Google Fact Check API and cross-referenced with authoritative sources via "
            f"Google Custom Search Engine. Results are categorized by verification status.",
            styles['DetailText']
        )
        elements.append(overview)
        elements.append(Spacer(1, 0.15*inch))
        
        # Summary table
        table_data = [["#", "Claim Summary", "Verification Method", "Status"]]
        
        for idx, claim_result in enumerate(claims, 1):
            claim_text = claim_result.get("claim", "")
            claim_summary = (claim_text[:50] + "...") if len(claim_text) > 50 else claim_text
            method = claim_result.get("method", "N/A")
            
            if method == "custom_search":
                method = "custom\nsearch"  # Forces line break
            elif method == "factcheck_api":
                method = "factcheck\napi"

            # Determine status
            fact_matches = claim_result.get("fact_matches", [])
            search_matches = claim_result.get("search_matches", [])
            
            if fact_matches:
                status = "Fact-checked"
            elif search_matches:
                status = "Cross-referenced"
            else:
                status = "Unverified"
            
            table_data.append([str(idx), claim_summary, method, status])
        
        claims_table = Table(table_data, colWidths=[0.4*inch, 3.5*inch, 1.3*inch, 1.3*inch])
        claims_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('WORDWRAP', (0, 0), (-1, -1)),
        ]))
        
        elements.append(claims_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements

    def build_pdf_factcheck_details(self, styles, pdf_claims_data: dict):
        """Build detailed fact-check results with proper page break control"""
        elements = []
        
        claims = pdf_claims_data.get("claims", [])
        
        if not claims:
            return elements
        
        # Section header
        header_elements = []
        header = Paragraph("Detailed Claim Verification", styles['SectionHeader'])
        header_elements.append(header)
        
        methodology = Paragraph(
            "<b>Verification Methodology:</b> Each claim undergoes two-tier verification: (1) <b>Fact Check API:</b> "
            "Searches Google's Fact Check database containing verified assessments from credible fact-checking organizations "
            "(e.g., PolitiFact, Snopes, FactCheck.org). (2) <b>Custom Search:</b> Queries authoritative sources to find "
            "corroborating or contradicting evidence. Claims are evaluated based on source credibility, recency, and "
            "consistency across multiple sources.",
            styles['DetailText']
        )
        header_elements.append(methodology)
        
        elements.append(KeepTogether(header_elements))
        elements.append(Spacer(1, 0.2*inch))
        
        # Process each claim
        for idx, claim_result in enumerate(claims, 1):
            # Add conditional page break - ensures claim doesn't start at page bottom
            elements.append(CondPageBreak(2.5*inch))
            
            claim_elements = []  # Collect all parts of this claim
            
            # Claim header with index
            claim_header = Paragraph(f"Claim #{idx}", styles['Subsection'])
            claim_elements.append(claim_header)
            
            # Full claim text
            claim_text = claim_result.get("claim", "N/A")
            claim_para = Paragraph(f"<b>Claim:</b> {claim_text}", styles['DetailText'])
            claim_elements.append(claim_para)
            claim_elements.append(Spacer(1, 0.12*inch))

            
            # Fact-check matches section
            fact_matches = claim_result.get("fact_matches", [])
            
            if fact_matches:
                fact_header = Paragraph("<b>Professional Fact-Check Results:</b>", styles['DetailText'])
                claim_elements.append(fact_header)
                claim_elements.append(Spacer(1, 0.08*inch))
                
                for match_idx, match in enumerate(fact_matches[:5], 1):  # Show up to 5
                    publisher = match.get("publisher", "Unknown Source")
                    rating = match.get("rating", "N/A")
                    title = match.get("title", "")
                    url = match.get("url", "")
                    claim_date = match.get("date", "Unknown date")
                    
                    match_text = f"<b>{match_idx}. {publisher}</b> (Published: {claim_date})<br/>"
                    match_text += f"   <b>Rating:</b> {rating}<br/>"
                    if title:
                        match_text += f"   <b>Title:</b> {title}<br/>"
                    if url:
                        match_text += f"   <b>Source:</b> <link href='{url}' color='blue'>{url[:70]}...</link><br/>"
                    
                    match_para = Paragraph(match_text, styles['DetailText'])
                    claim_elements.append(match_para)
                    claim_elements.append(Spacer(1, 0.08*inch))
                
                # Interpretation of fact-check ratings
                ratings_list = [m.get("rating", "").lower() for m in fact_matches]
                
                false_indicators = ["false", "incorrect", "pants on fire", "mostly false"]
                true_indicators = ["true", "correct", "accurate", "mostly true"]
                mixed_indicators = ["mixed", "half true", "partially true"]
                
                false_count = sum(1 for r in ratings_list if any(ind in r for ind in false_indicators))
                true_count = sum(1 for r in ratings_list if any(ind in r for ind in true_indicators))
                mixed_count = sum(1 for r in ratings_list if any(ind in r for ind in mixed_indicators))
                
                if false_count > 0:
                    interp = (
                        f"<b>Warning - Concern Identified:</b> {false_count} fact-checker(s) rated this claim as false or mostly false. "
                        "This suggests potential misinformation or inaccuracy. The claim should be treated with skepticism "
                        "and corrected if used in further content."
                    )
                    interp_color = colors.HexColor('#ffebee')
                elif mixed_count > 0 and true_count == 0:
                    interp = (
                        f"<b>Warning - Partial Accuracy:</b> {mixed_count} fact-checker(s) found mixed accuracy. The claim may contain "
                        "some truth but lacks full context, includes exaggerations, or oversimplifies complex issues. "
                        "Recommend adding nuance or clarification."
                    )
                    interp_color = colors.HexColor('#fff3e0')
                elif true_count > 0:
                    interp = (
                        f"<b>Verified Accurate:</b> {true_count} fact-checker(s) confirmed accuracy. This claim is supported "
                        "by credible sources and can be considered reliable based on available fact-checking evidence."
                    )
                    interp_color = colors.HexColor('#e8f5e9')
                else:
                    interp = (
                        "<b>Status Unclear:</b> Fact-check results are available but ratings are ambiguous. Manual review "
                        "of source material recommended."
                    )
                    interp_color = colors.HexColor('#f5f5f5')
                
                interp_para = Paragraph(interp, styles['DetailText'])
                interp_table = Table([[interp_para]], colWidths=[6.5*inch])
                interp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), interp_color),
                    ('BOX', (0, 0), (-1, -1), 2, colors.grey),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('WORDWRAP', (0, 0), (-1, -1)),
                ]))
                
                claim_elements.append(interp_table)
                claim_elements.append(Spacer(1, 0.15*inch))
            
            # Related sources section
            search_matches = claim_result.get("search_matches", [])
            
            if search_matches:
                search_header = Paragraph("<b>Supporting/Contradicting Sources:</b>", styles['DetailText'])
                claim_elements.append(search_header)
                claim_elements.append(Spacer(1, 0.08*inch))
                
                for src_idx, match in enumerate(search_matches[:5], 1):  # Show up to 5
                    title = match.get("title", "Unknown")
                    url = match.get("url", "")
                    snippet = match.get("snippet", "")
                    score = match.get("score", 0)
                    
                    source_text = f"<b>{src_idx}. {title}</b><br/>"
                    if snippet:
                        snippet_clean = snippet.replace('\n', ' ')[:200]
                        source_text += f"   <i>Excerpt:</i> {snippet_clean}...<br/>"
                    if url:
                        source_text += f"   <link href='{url}' color='blue'>{url[:70]}...</link><br/>"
                    source_text += f"   <i>Relevance score: {score:.2f}</i><br/>"
                    
                    source_para = Paragraph(source_text, styles['DetailText'])
                    claim_elements.append(source_para)
                    claim_elements.append(Spacer(1, 0.08*inch))
                
                # Guidance on interpreting search results
                search_guidance = Paragraph(
                    "<b>How to Use These Sources:</b> Cross-reference multiple sources to assess claim validity. "
                    "Prioritize recent publications from authoritative domains (.edu, .gov, established news organizations). "
                    "High relevance scores indicate semantic similarity but don't guarantee accuracy. Look for consensus "
                    "across independent sources and verify publication dates for time-sensitive claims.",
                    styles['DetailText']
                )
                claim_elements.append(search_guidance)
            
            # If no verification available
            if not fact_matches and not search_matches:
                no_verify = Paragraph(
                    "<b>Warning - No verification sources found.</b> This claim could not be verified through available fact-checking "
                    "databases or web search. This may indicate: (1) claim is too specific/obscure, (2) claim relates to "
                    "very recent events not yet fact-checked, (3) claim contains specialized knowledge outside mainstream sources, "
                    "or (4) potential hallucination with no basis in reality. <b>Recommend manual verification.</b>",
                    styles['DetailText']
                )
                
                no_verify_table = Table([[no_verify]], colWidths=[6.5*inch])
                no_verify_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff9c4')),
                    ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('WORDWRAP', (0, 0), (-1, -1)),
                ]))
                
                claim_elements.append(no_verify_table)
            
            # Wrap entire claim together to prevent splitting
            elements.append(KeepTogether(claim_elements))
            
            # Separator between claims
            if idx < len(claims):
                elements.append(Spacer(1, 0.15*inch))
                elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceBefore=0, spaceAfter=0))
                elements.append(Spacer(1, 0.2*inch))
            else:
                elements.append(Spacer(1, 0.3*inch))
        
        # ============ Overall Assessment ============
        overall_elements = []
        
        overall_header = Paragraph("Overall Factual Assessment", styles['Subsection'])
        overall_elements.append(overall_header)
        
        # Calculate overall statistics
        total_verified = sum(1 for c in claims if c.get("fact_matches"))
        total_unverified = len(claims) - total_verified
        
        assessment_text = (
            f"<b>Summary:</b> Of {len(claims)} claims analyzed, {total_verified} received professional fact-check "
            f"verification and {total_unverified} were cross-referenced via web search. "
        )
        
        # Count problematic ratings
        all_ratings = []
        for claim in claims:
            for match in claim.get("fact_matches", []):
                all_ratings.append(match.get("rating", "").lower())
        
        false_total = sum(1 for r in all_ratings if any(ind in r for ind in ["false", "incorrect", "pants on fire"]))
        
        if false_total > 0:
            assessment_text += (
                f"<b>⚠ {false_total} false rating(s) detected.</b> This indicates potential misinformation or factual "
                "errors in the text. AI-generated content frequently contains plausible-sounding but inaccurate statements "
                "('hallucinations'). These should be corrected before further use or distribution."
            )
            assessment_color = colors.HexColor('#ffebee')
        elif total_unverified == len(claims):
            assessment_text += (
                "<b>Note:</b> All claims remain unverified through fact-checking databases. This doesn't necessarily indicate "
                "falsehood, but suggests claims may be: too recent, too specific, or outside mainstream fact-checking scope. "
                "Independent verification recommended for critical applications."
            )
            assessment_color = colors.HexColor('#fff9c4')
        else:
            assessment_text += (
                "<b>✓ No false ratings detected.</b> Verified claims appear factually sound based on available fact-checking "
                "evidence. However, absence of false ratings doesn't guarantee complete accuracy, especially for unverified claims."
            )
            assessment_color = colors.HexColor('#e8f5e9')
        
        assessment_para = Paragraph(assessment_text, styles['DetailText'])
        assessment_table = Table([[assessment_para]], colWidths=[6.5*inch])
        assessment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), assessment_color),
            ('BOX', (0, 0), (-1, -1), 2, colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('WORDWRAP', (0, 0), (-1, -1)),
        ]))
        
        elements.append(assessment_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements

    def fastdetect_score(self, text, api_key):
        """
        Compute FastDetect score using the API
        """
        
        url = "https://api.fastdetect.net/api/detect"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = {
            "detector": "fast-detect(falcon-7b/falcon-7b-instruct)",
            "text": text
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = SimpleNamespace(**response.json())
            
            if result.code == 0:
                print(f"FastDetect api worked")
                return {
                    "success": True,
                    "prob": result.data.get("prob"),
                    "crit": result.data.get("details", {}).get("crit"),
                    "ntoken": result.data.get("details", {}).get("ntoken"),
                    "message": result.msg
                }
            else:
                return {
                    "success": False,
                    "error": result.msg,
                    "code": result.code
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def process_single_token(self, token):
        if token in ['<bos>', '</s>', '<pad>']:
            return f'⟨{token[1:-1]}⟩'
        if token.startswith('▁') or token.startswith('Ġ'):
            return ' ' + token[1:]
        return token
    
    def count_masks(self, texts):
        """Count number of mask tokens in texts"""
        if isinstance(texts, str):
            texts = [texts]
        return [len(self.EXTRA_ID_PATTERN.findall(text)) for text in texts]

    def extract_fills(self, texts):
        """Extract fill text between mask tokens"""
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        extracted_fills = [self.EXTRA_ID_PATTERN.split(x)[1:-1] for x in texts]
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
        return extracted_fills

    def apply_extracted_fills_robust(self, masked_texts, extracted_fills):
        """Apply fills back to masked text with robust error handling"""
        texts = []
        for i, masked_text in enumerate(masked_texts):
            n_expected = self.count_masks([masked_text])[0]
            fills = extracted_fills[i] if i < len(extracted_fills) else []
            
            if len(fills) < n_expected:
                texts.append("")
                continue
            
            temp_text = masked_text
            for fill_idx in range(n_expected):
                temp_text = temp_text.replace(f"<extra_id_{fill_idx}>", fills[fill_idx], 1)
            texts.append(temp_text)
        
        return texts

    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False, buffer_size=1):
        """Tokenize and mask text"""
        tokens = self.mask_tokenizer.tokenize(text)
        mask_string = '<<<mask>>>'
        
        n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)
        
        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - buffer_size)
            search_end = min(len(tokens), end + buffer_size)
            
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = self.mask_tokenizer.convert_tokens_to_string(tokens)
        return text

    def replace_masks(self, texts, mask_top_p=1.0):
        """Replace masks using the mask filling model - GPU optimized but keeping original logic"""
        n_expected = self.count_masks(texts)
        if not n_expected or max(n_expected) == 0:
            return texts
        
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        
        tokens = self.mask_tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            # Use CUDA mixed precision if available
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.mask_model.generate(
                        **tokens,
                        max_length=150,  # ← KEY: Use max_length like original
                        do_sample=True,
                        top_p=mask_top_p,  # ← Default 1.0 like original
                        num_return_sequences=1,
                        eos_token_id=stop_id
                    )
            else:
                outputs = self.mask_model.generate(
                    **tokens,
                    max_length=150,
                    do_sample=True,
                    top_p=mask_top_p,
                    num_return_sequences=1,
                    eos_token_id=stop_id
                )
        
        decoded = self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Cleanup
        del tokens, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return decoded

    def perturb_texts_single(self, texts, span_length, pct, ceil_pct=False, buffer_size=1):
        """Perturb a batch of texts - keeping original logic"""
        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct, buffer_size) 
                        for x in texts]
        
        # Simple validation
        valid_masks = [text for text in masked_texts if '<extra_id_' in text]
        if len(valid_masks) != len(masked_texts):
            print(f"WARNING: {len(masked_texts) - len(valid_masks)} texts had no valid masks")
        
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills_robust(masked_texts, extracted_fills)
        
        # Simple retry logic from original
        attempts = 1
        max_attempts = 5
        while '' in perturbed_texts and attempts < max_attempts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Retrying with less masking [attempt {attempts}].')
            
            retry_pct = max(0.1, pct - (0.05 * attempts))
            retry_span = max(1, span_length - (attempts // 2))
            
            masked_texts_retry = [self.tokenize_and_mask(texts[idx], retry_span, retry_pct, ceil_pct, buffer_size) 
                                for idx in idxs]
            raw_fills_retry = self.replace_masks(masked_texts_retry)
            extracted_fills_retry = self.extract_fills(raw_fills_retry)
            new_perturbed_texts = self.apply_extracted_fills_robust(masked_texts_retry, extracted_fills_retry)
            
            for idx, new_text in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = new_text
            
            attempts += 1
        
        # Final fallback
        for idx, text in enumerate(perturbed_texts):
            if text == '':
                print(f"WARNING: Text {idx} failed all perturbation attempts. Using original.")
                perturbed_texts[idx] = texts[idx]
        
        return perturbed_texts

    def perturb_text(self, text, span_length=2, pct=0.3, buffer_size=1):
        """Perturb a single text - keeping original logic"""
        try:
            tokens = self.tokenizer.encode(text)
            
            if len(tokens) > 400:
                print(f"Text is long ({len(tokens)} tokens). Using chunking approach...")
                return self.perturb_long_text(text, span_length, pct, buffer_size)
            
            # Normal perturbation
            perturbed = self.perturb_texts_single([text], span_length, pct, buffer_size=buffer_size)
            
            if perturbed and perturbed[0]:
                return perturbed[0]
            
            # Fallback
            print(f"Initial perturbation failed. Trying with reduced masking...")
            perturbed = self.perturb_texts_single([text], span_length=1, pct=0.15, buffer_size=buffer_size)
            
            return perturbed[0] if perturbed and perturbed[0] else text
            
        except Exception as e:
            print(f"Perturbation failed: {e}")
            return text

    def perturb_long_text(self, text, span_length, pct, buffer_size):
        """Perturb long text by chunking into sentences - original logic"""
        import re
        
        sentences = re.split(r'([.!?]+\s+)', text)
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return self.perturb_texts_single([text], span_length=1, pct=0.15, buffer_size=buffer_size)[0]
        
        perturbed_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                perturbed_sentences.append(sentence)
                continue
            
            try:
                perturbed = self.perturb_texts_single([sentence], span_length, pct, buffer_size=buffer_size)
                perturbed_sentences.append(perturbed[0] if perturbed and perturbed[0] else sentence)
            except:
                perturbed_sentences.append(sentence)
        
        return ''.join(perturbed_sentences)

    def get_ll(self, text):
        """Compute log-likelihood of text under the model"""
        try:
            with torch.no_grad():
                tokenized = self.tokenizer(text, return_tensors="pt", 
                                        max_length=512, truncation=True).to(self.device)
                labels = tokenized.input_ids
                outputs = self.model(**tokenized, labels=labels)
                return -outputs.loss.item()
        except Exception as e:
            print(f"Likelihood computation failed: {e}")
            return None

    def detectgpt_score(self, text, span_length=2, pct=0.3, n_perturbations=10, buffer_size=1):
        """
        Compute DetectGPT score using the exact methodology from run.py
        """
        try:
            torch.manual_seed(0)
            np.random.seed(0)
            
            print(f"Computing original likelihood...")
            original_ll = self.get_ll(text)
            if original_ll is None:
                return {"detectgpt_score": None, "error": "Failed to compute original likelihood"}
            
            print(f"Generating {n_perturbations} perturbations...")
            perturbed_texts = []
            for i in range(n_perturbations):
                pert = self.perturb_text(text, span_length, pct, buffer_size)
                if pert and pert != text and pert != "":
                    perturbed_texts.append(pert)
            
            if len(perturbed_texts) == 0:
                return {"detectgpt_score": None, "error": "All perturbations failed"}
            
            print(f"Computing likelihoods for {len(perturbed_texts)} perturbations...")
            perturbed_lls = []
            for pert_text in perturbed_texts:
                ll = self.get_ll(pert_text)
                if ll is not None:
                    perturbed_lls.append(ll)
            
            if len(perturbed_lls) == 0:
                return {"detectgpt_score": None, "error": "Failed to compute perturbed likelihoods"}
            
            perturbed_ll_mean = np.mean(perturbed_lls)
            perturbed_ll_std = np.std(perturbed_lls) if len(perturbed_lls) > 1 else 1
            print(f"original : {original_ll} and perturbed : {perturbed_ll_mean}")
            detectgpt_score = original_ll - perturbed_ll_mean
            
            print(f"DetectGPT score: {detectgpt_score:.4f}")
            
            return {
                "detectgpt_score": float(detectgpt_score),
                "original_ll": float(original_ll),
                "perturbed_ll_mean": float(perturbed_ll_mean),
                "perturbed_ll_std": float(perturbed_ll_std),
                "n_perturbations": len(perturbed_lls),
                "requested_perturbations": n_perturbations,
                "interpretation": "Negative scores suggest AI-generated text"
            }
        
        except Exception as e:
            print(f"DetectGPT computation failed: {e}")
            traceback.print_exc()
            return {"detectgpt_score": None, "error": str(e)}
    
    def postprocess(self, tokens):
        if isinstance(tokens, (list, tuple)) and all(isinstance(t, int) for t in tokens):
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        elif isinstance(tokens, int):
            decoded = self.tokenizer.decode([tokens], skip_special_tokens=True)
        else:
            decoded = str(tokens)
        
        tokenizer_type = type(self.tokenizer).__name__.lower()
        
        if 'gpt2' in tokenizer_type or 'roberta' in tokenizer_type:
            processed = (decoded
                         .replace('Ġ', ' ')
                         .replace('Ċ', '\n')
                         .replace('ĊĊ', '\n\n')
                         .replace('Ä', '')
                         .replace('ĄĊ', '\n')
                         .replace('ÃƒÂ¢', '"'))
        elif 'sentencepiece' in tokenizer_type or 't5' in tokenizer_type:
            processed = (decoded
                         .replace('▁', ' ')
                         .replace('<0x0A>', '\n'))
        else:
            processed = decoded
        
        processed = (processed
                     .replace('ĊĊ', '\n\n')
                     .replace('Ċ', '\n')
                     .replace('ĄĊ', '\n')
                     .replace('Ą', '')
                     .replace('Ä', '')
                     .replace('▁', ' ')
                     .replace('Ġ', ' ')
                     .replace('\u010A', '\n')
                     .replace('\u000A', '\n')
                     .replace('\r\n', '\n')
                     .replace('\r', '\n')
                     .replace('ÃƒÂ¢', '"')
                     .replace('Ã¢', '-')
                     .replace('Ã¢Ä¢', '-')
                     .replace('Ã¢â‚¬"', '–')
                     .replace('Ã¢â‚¬â„¢', "'")
                     .replace('Ã¢â‚¬Å"', '"')
                     .replace('Ã¢â‚¬', '"'))
        
        # Clean up spacing
        import re
        processed = re.sub(r'  +', ' ', processed)
        processed = re.sub(r'\n\n\n+', '\n\n', processed)
        
        return processed.strip()


def main():
    print("main works")

if __name__ == "__main__":
    main()