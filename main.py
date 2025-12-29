"""
Lease Signing Microservice
Detects DataMatrix codes in PDFs and places signature images at marked locations.

DataMatrix codes expected:
- TS_MAIN, TS_RIDER_A, TS_RIDER_B, TS_RIDER_C, TS_BEDBUG, TS_WIND (tenant signatures)
- LS_MAIN, LS_RIDER_A, LS_RIDER_B, LS_RIDER_C, LS_BEDBUG (landlord signatures)
"""

import io
import os
import uuid
import threading
import concurrent.futures
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from pdf2image import convert_from_bytes
from pylibdmtx.pylibdmtx import decode as decode_datamatrix
from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import logging
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lease Signer API",
    description="Detects DataMatrix signature markers in PDFs and applies signatures",
    version="1.2.0"
)

# In-memory job storage
jobs = {}

# Timeout for processing each page (seconds) - configurable via env var
PAGE_TIMEOUT = int(os.environ.get("PAGE_TIMEOUT", "60"))

# Signature code mappings
TENANT_SIGNATURE_CODES = {
    "TS_MAIN", "TS_RIDER_A", "TS_RIDER_B", "TS_RIDER_C", "TS_BEDBUG", "TS_WIND"
}
LANDLORD_SIGNATURE_CODES = {
    "LS_MAIN", "LS_RIDER_A", "LS_RIDER_B", "LS_RIDER_C", "LS_BEDBUG"
}
ALL_SIGNATURE_CODES = TENANT_SIGNATURE_CODES | LANDLORD_SIGNATURE_CODES


def find_datamatrix_codes(image: Image.Image, dpi: int = 200, page_height_points: float = 792) -> list[dict]:
    """
    Find and decode all DataMatrix codes in an image.
    Returns list of dicts with 'data', 'x', 'y', 'width', 'height' in PDF points.
    
    Note: PDF coordinates have origin at bottom-left, but we need to place
    signatures correctly, so we convert from image coords (top-left origin).
    """
    decoded = decode_datamatrix(image)
    
    results = []
    for dm in decoded:
        scale = 72.0 / dpi
        
        pixel_left = dm.rect.left
        pixel_top = dm.rect.top
        pixel_width = dm.rect.width
        pixel_height = dm.rect.height
        
        # Convert to PDF coordinates
        # Image: origin at top-left, y increases downward
        # PDF: origin at bottom-left, y increases upward
        pdf_x = pixel_left * scale
        
        # Convert image y (from top) to PDF y (from bottom)
        # PDF_y = page_height - image_y_in_points
        image_y_points = pixel_top * scale
        pdf_y = page_height_points - image_y_points - (pixel_height * scale)
        
        pdf_width = pixel_width * scale
        pdf_height = pixel_height * scale
        
        try:
            data = dm.data.decode('utf-8').strip()
        except:
            data = str(dm.data)
        
        results.append({
            'data': data,
            'x': pdf_x,
            'y': pdf_y,
            'width': pdf_width,
            'height': pdf_height,
        })
        logger.info(f"Found DataMatrix: '{data}' at PDF coords ({pdf_x:.1f}, {pdf_y:.1f})")
    
    return results


def convert_single_page(pdf_bytes: bytes, page_num: int, dpi: int = 200) -> Image.Image:
    """Convert a single page of a PDF to an image."""
    images = convert_from_bytes(
        pdf_bytes, 
        dpi=dpi, 
        first_page=page_num, 
        last_page=page_num
    )
    return images[0] if images else None


def process_page_with_timeout(pdf_bytes: bytes, page_num: int, dpi: int, page_height_points: float = 792, timeout: int = PAGE_TIMEOUT):
    """Process a single page with a timeout. Returns (image, codes) or (None, []) on timeout."""
    def do_work():
        image = convert_single_page(pdf_bytes, page_num, dpi)
        if image:
            codes = find_datamatrix_codes(image, dpi=dpi, page_height_points=page_height_points)
            return image, codes
        return None, []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(do_work)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Page {page_num} timed out after {timeout}s")
            return None, []


def create_signature_overlay(
    page_width: float,
    page_height: float,
    signatures_to_place: list[dict],
    tenant_sig_image: Optional[bytes],
    landlord_sig_image: Optional[bytes],
    sig_width: float = 100,
    sig_height: float = 40,
    x_offset: float = 30,
    y_offset: float = 0
) -> bytes:
    """Create a PDF page with signature images placed at specified locations."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    
    for sig_info in signatures_to_place:
        code = sig_info['data']
        
        if code in TENANT_SIGNATURE_CODES and tenant_sig_image:
            sig_bytes = tenant_sig_image
        elif code in LANDLORD_SIGNATURE_CODES and landlord_sig_image:
            sig_bytes = landlord_sig_image
        else:
            continue
        
        sig_x = sig_info['x'] + x_offset
        sig_y = sig_info['y'] + y_offset
        
        try:
            sig_image = ImageReader(io.BytesIO(sig_bytes))
            c.drawImage(
                sig_image,
                sig_x,
                sig_y,
                width=sig_width,
                height=sig_height,
                preserveAspectRatio=True,
                mask='auto'
            )
            logger.info(f"Placed signature for {code} at ({sig_x:.1f}, {sig_y:.1f})")
        except Exception as e:
            logger.error(f"Failed to place signature for {code}: {e}")
    
    c.save()
    buffer.seek(0)
    return buffer.read()


def merge_overlay_with_page(original_page, overlay_pdf_bytes) -> None:
    """Merge an overlay PDF onto an original page."""
    overlay_reader = PdfReader(io.BytesIO(overlay_pdf_bytes))
    if len(overlay_reader.pages) > 0:
        original_page.merge_page(overlay_reader.pages[0])


def process_scan_job(job_id: str, pdf_bytes: bytes, dpi: int):
    """Background job to scan PDF for DataMatrix codes."""
    try:
        jobs[job_id]['status'] = 'processing'
        
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        
        all_codes = []
        skipped_pages = []
        
        for page_num in range(1, total_pages + 1):
            jobs[job_id]['progress'] = f"Scanning page {page_num}/{total_pages}"
            logger.info(f"Job {job_id}: Scanning page {page_num}/{total_pages}")
            
            # Get the actual page height
            page = pdf_reader.pages[page_num - 1]
            page_height = float(page.mediabox.height)
            
            image, codes = process_page_with_timeout(pdf_bytes, page_num, dpi, page_height_points=page_height)
            
            if image is None and not codes:
                logger.warning(f"Job {job_id}: Page {page_num} skipped (timeout or error)")
                skipped_pages.append(page_num)
            else:
                for code in codes:
                    code['page'] = page_num
                    all_codes.append(code)
                if image:
                    del image
        
        signature_codes = [c for c in all_codes if c['data'] in ALL_SIGNATURE_CODES]
        metadata_codes = [c for c in all_codes if c['data'] not in ALL_SIGNATURE_CODES]
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {
            "total_pages": total_pages,
            "total_codes_found": len(all_codes),
            "signature_markers": signature_codes,
            "metadata_codes": metadata_codes,
            "skipped_pages": skipped_pages
        }
        logger.info(f"Job {job_id}: Scan completed. Found {len(all_codes)} codes. Skipped pages: {skipped_pages}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)


def process_sign_job(
    job_id: str,
    pdf_bytes: bytes,
    tenant_sig_bytes: Optional[bytes],
    landlord_sig_bytes: Optional[bytes],
    dpi: int,
    sig_width: float,
    sig_height: float,
    x_offset: float,
    y_offset: float
):
    """Background job to sign PDF."""
    try:
        jobs[job_id]['status'] = 'processing'
        
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        pdf_writer = PdfWriter()
        total_pages = len(pdf_reader.pages)
        
        signatures_applied = []
        all_codes_found = []
        skipped_pages = []
        
        for page_num in range(1, total_pages + 1):
            jobs[job_id]['progress'] = f"Processing page {page_num}/{total_pages}"
            logger.info(f"Job {job_id}: Processing page {page_num}/{total_pages}")
            
            original_page = pdf_reader.pages[page_num - 1]
            page_width = float(original_page.mediabox.width)
            page_height = float(original_page.mediabox.height)
            
            image, codes = process_page_with_timeout(pdf_bytes, page_num, dpi, page_height_points=page_height)
            
            if image is None and not codes:
                logger.warning(f"Job {job_id}: Page {page_num} skipped (timeout or error)")
                skipped_pages.append(page_num)
            else:
                if image:
                    del image  # Free memory immediately
                
                for code in codes:
                    code['page'] = page_num
                    all_codes_found.append(code)
                
                sig_codes = [c for c in codes if c['data'] in ALL_SIGNATURE_CODES]
                
                if sig_codes:
                    overlay_bytes = create_signature_overlay(
                        page_width=page_width,
                        page_height=page_height,
                        signatures_to_place=sig_codes,
                        tenant_sig_image=tenant_sig_bytes,
                        landlord_sig_image=landlord_sig_bytes,
                        sig_width=sig_width,
                        sig_height=sig_height,
                        x_offset=x_offset,
                        y_offset=y_offset
                    )
                    merge_overlay_with_page(original_page, overlay_bytes)
                    
                    for sc in sig_codes:
                        signatures_applied.append({
                            'page': page_num,
                            'code': sc['data'],
                            'x': round(sc['x'], 2),
                            'y': round(sc['y'], 2)
                        })
            
            pdf_writer.add_page(original_page)
        
        output_buffer = io.BytesIO()
        pdf_writer.write(output_buffer)
        output_buffer.seek(0)
        pdf_base64 = base64.b64encode(output_buffer.read()).decode('utf-8')
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {
            "success": True,
            "total_pages": total_pages,
            "signatures_applied": signatures_applied,
            "signatures_count": len(signatures_applied),
            "all_codes_found": [
                {"page": c['page'], "data": c['data'], "x": round(c['x'], 2), "y": round(c['y'], 2)}
                for c in all_codes_found
            ],
            "skipped_pages": skipped_pages,
            "pdf_base64": pdf_base64
        }
        logger.info(f"Job {job_id}: Signing completed. Applied {len(signatures_applied)} signatures. Skipped: {skipped_pages}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "lease-signer-api", "version": "1.2.0"}


@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    """Check the status of a background job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job['status'],
        "progress": job.get('progress', '')
    }
    
    if job['status'] == 'completed':
        # Return result without the large pdf_base64 field
        result_copy = {k: v for k, v in job['result'].items() if k != 'pdf_base64'}
        result_copy['has_pdf'] = 'pdf_base64' in job['result']
        response['result'] = result_copy
    elif job['status'] == 'failed':
        response['error'] = job.get('error', 'Unknown error')
    
    return response


@app.get("/job/{job_id}/download")
def download_signed_pdf(job_id: str):
    """Download the signed PDF from a completed job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(400, f"Job not completed. Current status: {job['status']}")
    
    if 'pdf_base64' not in job.get('result', {}):
        raise HTTPException(400, "This job does not contain a PDF (scan-only job)")
    
    pdf_bytes = base64.b64decode(job['result']['pdf_base64'])
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=signed_{job_id[:8]}.pdf"
        }
    )


@app.post("/scan")
async def scan_pdf_for_codes(
    pdf_file: UploadFile = File(...),
    dpi: int = Form(default=150)
):
    """
    Scan a PDF and return all DataMatrix codes found.
    Returns a job_id - poll /job/{job_id} to get results.
    
    Note: Uses lower default DPI (150) for faster processing.
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await pdf_file.read()
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'queued',
        'progress': 'Starting...',
        'result': None
    }
    
    thread = threading.Thread(target=process_scan_job, args=(job_id, pdf_bytes, dpi))
    thread.start()
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Poll /job/{job_id} to check status and get results"
    }


@app.post("/sign")
async def sign_pdf(
    pdf_file: UploadFile = File(...),
    tenant_signature: Optional[UploadFile] = File(default=None),
    landlord_signature: Optional[UploadFile] = File(default=None),
    dpi: int = Form(default=150),
    sig_width: float = Form(default=100),
    sig_height: float = Form(default=40),
    x_offset: float = Form(default=30),
    y_offset: float = Form(default=0)
):
    """
    Sign a PDF by detecting DataMatrix markers and placing signatures.
    Returns a job_id - poll /job/{job_id} to get results including base64 PDF.
    
    Note: Uses lower default DPI (150) for faster processing.
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await pdf_file.read()
    
    tenant_sig_bytes = None
    landlord_sig_bytes = None
    
    if tenant_signature:
        tenant_sig_bytes = await tenant_signature.read()
    if landlord_signature:
        landlord_sig_bytes = await landlord_signature.read()
    
    if not tenant_sig_bytes and not landlord_sig_bytes:
        raise HTTPException(400, "At least one signature image must be provided")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'queued',
        'progress': 'Starting...',
        'result': None
    }
    
    thread = threading.Thread(
        target=process_sign_job,
        args=(job_id, pdf_bytes, tenant_sig_bytes, landlord_sig_bytes, dpi, sig_width, sig_height, x_offset, y_offset)
    )
    thread.start()
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Poll /job/{job_id} to check status and get results"
    }


@app.post("/scan-sync")
async def scan_pdf_sync(
    pdf_file: UploadFile = File(...),
    dpi: int = Form(default=150)
):
    """
    Synchronous scan - use for small PDFs only (< 5 pages).
    For larger PDFs, use /scan which runs in background.
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await pdf_file.read()
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(pdf_reader.pages)
    
    if total_pages > 5:
        raise HTTPException(400, f"PDF has {total_pages} pages. Use /scan endpoint for PDFs with more than 5 pages.")
    
    all_codes = []
    
    for page_num in range(1, total_pages + 1):
        page = pdf_reader.pages[page_num - 1]
        page_height = float(page.mediabox.height)
        
        image = convert_single_page(pdf_bytes, page_num, dpi)
        if image:
            codes = find_datamatrix_codes(image, dpi=dpi, page_height_points=page_height)
            for code in codes:
                code['page'] = page_num
                all_codes.append(code)
            del image
    
    signature_codes = [c for c in all_codes if c['data'] in ALL_SIGNATURE_CODES]
    metadata_codes = [c for c in all_codes if c['data'] not in ALL_SIGNATURE_CODES]
    
    return {
        "total_pages": total_pages,
        "total_codes_found": len(all_codes),
        "signature_markers": signature_codes,
        "metadata_codes": metadata_codes
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
