# # app/main.py
# import io
# import re
# import math
# import tempfile
# import requests
# from typing import List, Dict, Any
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from pdf2image import convert_from_bytes
# from PIL import Image
# import pytesseract
# from collections import defaultdict
# from decimal import Decimal, ROUND_HALF_UP
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from google import genai
# import os
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# app = FastAPI(title="Invoice Line-Item Extractor")
# amount_re = re.compile(r'([0-9]{1,3}(?:[,0-9]{0,3})*(?:\.\d{1,2})?)') 
# qty_re = re.compile(r'Qty[:\s]*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
# rate_re = re.compile(r'Rate[:\s]*([0-9]{1,3}(?:[,0-9]{0,3})*(?:\.\d{1,2})?)', re.IGNORECASE)

# genai.configure(api_key=os.getenv("AIzaSyBNVfbH89hwYj_3ttfZklV1xmUI2I1BJzk"))
# client = genai.Client()

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)

# def gemini_parse_bill_text(text: str) -> List[Dict[str, Any]]:
#     prompt = f"""
# You are an expert at parsing invoices and bills.
# Extract all line items from the bill text. For each line item, output:
# - item_name (exactly as in bill)
# - item_quantity (if present)
# - item_rate (if present)
# - item_amount (net amount)

# Respond only with JSON: a list of objects.
# Bill text:
# {text}
# """
#     response = client.models.generate_content(
#         model="gemini-2.5-flash", 
#         contents=prompt
#     )
#     # The SDK returns response.text
#     import json
#     try:
#         data = json.loads(response.text)
#     except json.JSONDecodeError:
#         # optionally fallback or handle parse error
#         data = []
#     return data


# def safe_decimal(s):
#     if s is None: return None
#     s = s.replace(',', '')
#     try:
#         return float(Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
#     except Exception:
#         try:
#             return float(s)
#         except:
#             return None

# def download_doc(url: str) -> bytes:
#     resp = requests.get(url, timeout=30)
#     if resp.status_code != 200:
#         raise HTTPException(status_code=400, detail=f"Failed to download document: {resp.status_code}")
#     return resp.content

# def ocr_image(img: Image.Image) -> str:
#     return pytesseract.image_to_string(img, lang='eng')

# def find_amounts_in_line(line: str) -> List[float]:
#     res = []
#     for m in amount_re.findall(line):
#         val = safe_decimal(m)
#         if val is not None:
#             res.append(val)
#     return res

# def parse_page_text_to_items(page_text: str) -> List[Dict[str, Any]]:
#     lines = [l.strip() for l in page_text.splitlines() if l.strip()]
#     items = []

#     for line in lines:
#         original_line = line
#         norm = line.lower()
#         qty = None
#         rate = None
#         amount = None
#         m_qty = re.search(r"(qty|quantity)[:\s]*([0-9]+(?:\.[0-9]+)?)", norm)
#         if m_qty:
#             qty = safe_decimal(m_qty.group(2))
#         m_rate = re.search(r"(rate|price)[:\s]*([0-9]{1,7}(?:\.[0-9]+)?)", norm)
#         if m_rate:
#             rate = safe_decimal(m_rate.group(2))
#         m_amt = re.search(r"(amount|amt|total)[:\s]*([0-9]{1,10}(?:\.[0-9]+)?)", norm)
#         if m_amt:
#             amount = safe_decimal(m_amt.group(2))
#         m_mult = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[xX\*]\s*([0-9]+(?:\.[0-9]+)?)", norm)
#         if m_mult:
#             qty = qty or safe_decimal(m_mult.group(1))
#             rate = rate or safe_decimal(m_mult.group(2))
#             if qty and rate:
#                 amount = amount or qty * rate
#         numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", line)

#         if len(numbers) == 3:
#             q, r, a = numbers
#             qty = qty or safe_decimal(q)
#             rate = rate or safe_decimal(r)
#             amount = amount or safe_decimal(a)

#         if len(numbers) == 2:
#             r, a = numbers
#             rate = rate or safe_decimal(r)
#             amount = amount or safe_decimal(a)

#         if len(numbers) == 1:
#             if amount is None:
#                 amount = safe_decimal(numbers[0])
#         name = re.sub(r"[0-9\.\,\s]+$", "", original_line).strip()
#         if not name:
#             name = original_line
#         if amount is not None:
#             items.append({
#                 "item_name": name,
#                 "item_amount": float(amount) if amount else None,
#                 "item_rate": float(rate) if rate else None,
#                 "item_quantity": float(qty) if qty else None,
#             })
#     return items
# parsed_items = parse_page_text_to_items(text)
# if len(parsed_items) < MIN_THRESHOLD:
#     parsed_items = gemini_parse_bill_text(text)

# def deduplicate_items(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
#     """
#     Simple dedupe: group by (name normalized, amount, rate, qty) using fuzzy normalization.
#     """
#     grouped = {}
#     def normalize_name(n):
#         return re.sub(r'[^a-z0-9]', '', (n or '').lower())

#     for it in items:
#         key = (normalize_name(it.get("item_name")), it.get("item_amount"), it.get("item_rate"), it.get("item_quantity"))
#         if key in grouped:
#             existing = grouped[key]
#             for f in ("item_amount","item_rate","item_quantity"):
#                 if existing.get(f) is None and it.get(f) is not None:
#                     existing[f] = it[f]
#         else:
#             grouped[key] = dict(it)
#     return list(grouped.values())
# class RequestBody(BaseModel):
#     document: str



# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# @app.post("/extract-bill-data")
# def extract_bill_data(req: RequestBody):
#     try:
#         raw = download_doc(req.document)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     images = []
#     try:
#         images = convert_from_bytes(raw) 
#     except Exception:
#         try:
#             img = Image.open(io.BytesIO(raw)).convert("RGB")
#             images = [img]
#         except Exception as e:
#             raise HTTPException(status_code=400, detail="Unsupported document type or corrupt file.")

#     pagewise = []
#     all_items = []
#     for idx, img in enumerate(images, start=1):
#         text = ocr_image(img)
#         parsed_items = parse_page_text_to_items(text)
#         deduped = deduplicate_items(parsed_items)
#         pagewise.append({
#             "page_no": str(idx),
#             "page_type": "Bill Detail",
#             "bill_items": [
#                 {
#                     "item_name": it.get("item_name"),
#                     "item_amount": it.get("item_amount"),
#                     "item_rate": it.get("item_rate"),
#                     "item_quantity": it.get("item_quantity")
#                 } for it in deduped
#             ]
#         })
#         all_items.extend(deduped)

#     total_item_count = len(all_items)

#     final_total = sum(it.get("item_amount", 0) or 0 for it in all_items)
#     final_total = float(Decimal(final_total).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

#     response = {
#         "is_success": True,
#         "token_usage": {  
#             "total_tokens": 0,
#             "input_tokens": 0,
#             "output_tokens": 0
#         },
#         "data": {
#             "pagewise_line_items": pagewise,
#             "total_item_count": total_item_count,
#             "reconciled_amount": final_total
#         }
#     }
#     return response
import io
import re
import math
import tempfile
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import google.generativeai as genai
genai.configure(api_key="AIzaSyBNVfbH89hwYj_3ttfZklV1xmUI2I1BJzk")

GEMINI_VISION = genai.GenerativeModel("gemini-1.5-flash")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
app = FastAPI()

# -------------------------
# FIX CORS
# -------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# OCR / PDF helpers
# -------------------------

def safe_decimal(s):
    if s is None:
        return None
    s = str(s).replace(",", "")
    try:
        return float(Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except:
        return None

def download_doc(url: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise HTTPException(400, f"Failed to download document: {r.status_code}")
    return r.content

# -------------------------
# ⭐ Gemini Page Type Classification
# -------------------------
def classify_page_type(img: Image.Image) -> str:
    prompt = """
    You are a document classifier. Classify this page into one of these:
    - Bill Detail
    - Final Bill
    - Pharmacy

    Return ONLY the label.
    """
    try:
        resp = GEMINI_VISION.generate_content(
            [prompt, img],
            max_output_tokens=20
        )
        label = resp.text.strip()
        if label not in ["Bill Detail", "Final Bill", "Pharmacy"]:
            return "Bill Detail"
        return label
    except:
        return "Bill Detail"

# -------------------------
# ⭐ Gemini Line-Item Extractor
# -------------------------
def gemini_extract_items(img: Image.Image) -> List[Dict[str, Any]]:
    prompt = """
    Extract line items from this invoice page.
    Return ONLY JSON array of objects in this EXACT format:

    [
      {
        "item_name": "string",
        "item_quantity": float or null,
        "item_rate": float or null,
        "item_amount": float or null
      }
    ]

    Rules:
    - item_name must be EXACT text from invoice.
    - Use net amount (post-discount).
    - Convert all numbers to float.
    - If unclear, set field to null.
    """
    
    try:
        resp = GEMINI_VISION.generate_content(
            [prompt, img],
            response_mime_type="application/json"
        )
        return resp.parsed
    except:
        return []  # Gemini failed → fallback

# -------------------------
# ⭐ Improved Tesseract Fallback Extractor
# -------------------------
def ocr_fallback_items(page_text: str) -> List[Dict[str, Any]]:
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    items = []

    for line in lines:
        original_line = line
        lower = line.lower()

        qty = None
        rate = None
        amount = None

        # Qty/Rate/Amount
        m_qty = re.search(r"(qty|quantity)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", lower)
        if m_qty:
            qty = safe_decimal(m_qty.group(2))

        m_rate = re.search(r"(rate|price)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", lower)
        if m_rate:
            rate = safe_decimal(m_rate.group(2))

        m_amount = re.search(r"(amount|amt|total)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", lower)
        if m_amount:
            amount = safe_decimal(m_amount.group(2))

        # Detect "3 x 200"
        m_mult = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[xX\*]\s*([0-9]+(?:\.[0-9]+)?)", lower)
        if m_mult:
            qty = qty or safe_decimal(m_mult.group(1))
            rate = rate or safe_decimal(m_mult.group(2))
            if qty and rate:
                amount = amount or qty * rate

        # Column detection: qty rate amount
        nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", line)
        if len(nums) == 3:
            qty = qty or safe_decimal(nums[0])
            rate = rate or safe_decimal(nums[1])
            amount = amount or safe_decimal(nums[2])
        elif len(nums) == 2:
            rate = rate or safe_decimal(nums[0])
            amount = amount or safe_decimal(nums[1])
        elif len(nums) == 1 and amount is None:
            amount = safe_decimal(nums[0])

        name = re.sub(r"[0-9\.\,\s]+$", "", original_line).strip()
        if not name:
            name = original_line

        if amount:
            items.append({
                "item_name": name,
                "item_quantity": qty,
                "item_rate": rate,
                "item_amount": amount
            })
    
    return items

# -------------------------
# Dedupe
# -------------------------
def dedupe(items):
    out = {}
    for it in items:
        key = (it["item_name"].lower(), it["item_amount"])
        out[key] = it
    return list(out.values())

# -------------------------
# Request Model
# -------------------------
class Req(BaseModel):
    document: str

# -------------------------
# MAIN API
# -------------------------
@app.post("/extract-bill-data")
def extract(req: Req):

    raw = download_doc(req.document)

    try:
        pages = convert_from_bytes(raw)
    except:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        pages = [img]

    all_items = []
    pagewise = []

    for idx, img in enumerate(pages, 1):

        # 1. Classify page
        page_type = classify_page_type(img)

        # 2. Try Gemini extraction
        items = gemini_extract_items(img)

        # 3. If Gemini fails → Tesseract fallback
        if not items:
            text = pytesseract.image_to_string(img)
            items = ocr_fallback_items(text)

        # 4. Deduplicate
        items = dedupe(items)

        pagewise.append({
            "page_no": str(idx),
            "page_type": page_type,
            "bill_items": items
        })

        all_items.extend(items)

    total_item_count = len(all_items)
    final_total = sum([it["item_amount"] or 0 for it in all_items])

    return {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "data": {
            "pagewise_line_items": pagewise,
            "total_item_count": total_item_count,
            "reconciled_amount": final_total
        }
    }
