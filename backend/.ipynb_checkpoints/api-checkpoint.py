import token
import numpy as np
import torch
import time
import os
from dotenv import load_dotenv
import traceback
from accelerate import init_empty_weights, infer_auto_device_map

from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mxfp4Config, MBart50Tokenizer, MBartForConditionalGeneration

from .class_register import register_api

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(PROJECT_ROOT, ".env")

load_dotenv(dotenv_path)
auth = os.getenv("auth")

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
     return torch.where(logits < min_values,torch.ones_like(logits, dtype=logits.dtype) * -1e10,logits)

@register_api(name='gpt-oss-20b')
class GPTOSS20BChecker:
    def __init__(self, model_name_or_path="openai/gpt-oss-20b", hfauth=None, topk=40):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.topk = topk

        print(f"Loading {model_name_or_path} on {self.device} with 8-bit quantization...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            token=hfauth
        )
        print("Loaded slow tokenizer.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        last_exc = None
        try:
            # ✅ Define quant config properly
            # quant_config = Mxfp4Config(
            #     dtype=torch.float16,  # ✅ Force Half precision (fixes the error)
            #     quant_method="mxfp4",
            #     load_in_8bit=True,
            # )
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     model_name_or_path,
            #     quantization_config=quant_config,  # ✅ Correct way to pass quantization
            #     device_map="auto",
            #     token=hfauth,
            #     trust_remote_code=True,
            # )
              quant_config = Mxfp4Config(
                dtype=torch.float16,
                quant_method="mxfp4",
                # Remove load_in_8bit=True, try without it for potentially lower memory
            )
            
              self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=quant_config,
                device_map="auto",
                token=hfauth,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "8GB",1: "10GB",2: "10GB",3: "15GB"}
            )

              print("✅ Loaded model with Mxfp4 quantization config and device_map='auto'")

        except Exception as e:
            last_exc = e
            print("❌ Loading with Mxfp4Config quantization failed:", e)

        if not hasattr(self, "model"):
            print("🚨 Model failed to load. See last exception below:\n")
            traceback.print_exception(type(last_exc), last_exc, last_exc.__traceback__)
            raise last_exc
        self.mask_tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
        self.mask_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50", dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)    
        self.model.eval()
        print(f"🚀 Model ready on {self.device}")

    def check_probabilities(self, in_text, topk=40, include_detectgpt=True, include_fastdetect=True, fastdetect_api_key=None):
        # Tokenize input
        inputs = self.tokenizer(in_text, return_tensors="pt").to(self.device)
        token_ids = inputs["input_ids"][0]
        # Forward pass
        with torch.no_grad():
            logits = self.model(token_ids.unsqueeze(0)).logits
        all_probs = torch.softmax(logits[0, :-1, :], dim=-1)
        y = token_ids[1:]
    
        # Real token positions
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
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
    
        # Compute DetectGPT score if requested
        detectgpt_result = None
        if include_detectgpt:
            print("Computing DetectGPT score...")
            detectgpt_result = self.detectgpt_score(in_text)
    
        # Compute FastDetect score if requested
        fastdetect_result = None
        if include_fastdetect and fastdetect_api_key:
            print("Computing FastDetect score...")
            fastdetect_result = self.fastdetect_score(in_text, fastdetect_api_key)
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
    
        return result
    
    def fastdetect_score(self, text, api_key):
        """
        Compute FastDetect score using the API
        """
        import requests
        from types import SimpleNamespace
        
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
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            result = SimpleNamespace(**response.json())
            
            if result.code == 0:  # Success
                return {
                    "success": True,
                    "prob": result.data.get("prob"),
                    "crit": result.data.get("details", {}).get("crit"),
                    "ntoken": result.data.get("details", {}).get("ntoken"),
                    "message": result.msg
                }
            else:  # Error
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
    def get_ll(self, text):
        """Compute negative log-likelihood for a given text"""
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            labels = tokenized.input_ids
            outputs = self.model(**tokenized, labels=labels)
            return -outputs.loss.item()

    def perturb_text(self, text, span_length=2, pct=0.3):
        """Perturb input text by masking and filling"""
        # Tokenize & mask
        tokens = self.mask_tokenizer.tokenize(text)
        n_spans = int(pct * len(tokens) / (span_length + 2))
        for _ in range(n_spans):
            start = np.random.randint(0, max(1, len(tokens) - span_length))
            end = start + span_length
            tokens[start:end] = ["<mask>"]  # Use proper mask token
        masked_text = self.mask_tokenizer.convert_tokens_to_string(tokens)
    
        # Replace mask with fill
        self.mask_tokenizer.src_lang = "en_XX"
        input_tokens = self.mask_tokenizer(masked_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.mask_model.generate(**input_tokens, max_length=100, do_sample=True)
        filled = self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return filled[0]
    
    def detectgpt_score(self, text, span_length=2, pct=0.3, n_perturbations=10):
        """Compute DetectGPT score for a text"""
        try:
            # Original likelihood
            orig_ll = self.get_ll(text)
        
            # Perturbed likelihoods
            pert_lls = []
            for _ in range(n_perturbations):
                pert = self.perturb_text(text, span_length, pct)
                pert_lls.append(self.get_ll(pert))
        
            avg_pert_ll = np.mean(pert_lls)
            score = orig_ll - avg_pert_ll
            
            return {
                "detectgpt_score": float(score),
                "original_ll": float(orig_ll),
                "avg_perturbed_ll": float(avg_pert_ll),
                "n_perturbations": n_perturbations
            }
        except Exception as e:
            print(f"DetectGPT computation failed: {e}")
            return {
                "detectgpt_score": None,
                "original_ll": None,
                "avg_perturbed_ll": None,
                "n_perturbations": 0,
                "error": str(e)
            }
    
    def postprocess(self, tokens):
        if isinstance(tokens, (list, tuple)) and all(isinstance(t, int) for t in tokens):
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        elif isinstance(tokens, int):
            decoded = self.tokenizer.decode([tokens], skip_special_tokens=True)
        else:
            decoded = str(tokens)
        
        # Get tokenizer type for specific handling
        tokenizer_type = type(self.tokenizer).__name__.lower()
        
        if 'gpt2' in tokenizer_type or 'roberta' in tokenizer_type:
            # GPT-2/RoBERTa specific fixes
            processed = (decoded
                         .replace('Ġ', ' ')
                         .replace('Ċ', '\n')
                         .replace('ĊĊ', '\n\n')
                         .replace('Ä', '')
                         .replace('ÄŠ', '\n')
                         .replace('Ã¢', '"'))
        elif 'sentencepiece' in tokenizer_type or 't5' in tokenizer_type:
            # SentencePiece specific fixes
            processed = (decoded
                         .replace('▁', ' ')
                         .replace('<0x0A>', '\n'))
        else:
            # Generic fixes
            processed = decoded
        
        # Universal fixes for common issues
        processed = (processed
                    
                     .replace('ĊĊ', '\n\n')      # Double newline FIRST
                     .replace('Ċ', '\n')         # Single newline AFTER
                     .replace('ÄŠ', '\n')        # Alternative newline encoding
                     .replace('Ä ', '')          # Space prefix artifact
                     .replace('Ä', '')           # General artifact
                     .replace('▁', ' ')          # SentencePiece space marker
                     .replace('Ġ', ' ')          # GPT-2/RoBERTa space marker  
                     .replace('\u010A', '\n')    # Unicode newline (LF)
                     .replace('\u000A', '\n')    # Another newline encoding
                     .replace('\r\n', '\n')      # Windows line endings
                     .replace('\r', '\n')        # Mac line endings
                     .replace('Ã¢', '"')         # Common encoding artifact for quotes
                     .replace('â', '-')          # Common hyphen/dash encoding
                     .replace('âĢ', '-')         # En dash variant
                     .replace('â€"', '—')        # Em dash
                     .replace('â€™', "'")        # Right single quotation mark
                     .replace('â€œ', '"')        # Left double quotation mark  
                     .replace('â€', '"'))         
        
        # Clean up spacing
        import re
        processed = re.sub(r'  +', ' ', processed)
        processed = re.sub(r'\n\n\n+', '\n\n', processed)
        
        return processed.strip()
        

def main():
    print("main works")

if __name__ == "__main__":
    main()
