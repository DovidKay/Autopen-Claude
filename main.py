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
import json
import threading
import concurrent.futures
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response, HTMLResponse
from pdf2image import convert_from_bytes
from pylibdmtx.pylibdmtx import decode as decode_datamatrix
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import logging
import base64
import time

# Import signature detection module
from signature_detection import process_page_for_signatures, detect_if_phone_scan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lease Signer API",
    description="Detects DataMatrix signature markers in PDFs and applies signatures",
    version="1.3.0"
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
    
    Note: pylibdmtx returns coordinates where rect.top is measured from the
    BOTTOM of the image (like PDF coordinates), not from the top like typical
    image coordinates.
    """
    decoded = decode_datamatrix(image)
    
    scale = 72.0 / dpi
    logger.info(f"Image size: {image.width}x{image.height} pixels, DPI: {dpi}, scale: {scale}, page_height_points: {page_height_points}")
    
    results = []
    for dm in decoded:
        pixel_left = dm.rect.left
        pixel_top = dm.rect.top  # This is actually from BOTTOM of image
        pixel_width = dm.rect.width
        pixel_height = dm.rect.height
        
        # Convert to PDF coordinates
        # pylibdmtx rect.top is from bottom of image, same as PDF convention
        # So we can use it directly (just scale to points)
        pdf_x = pixel_left * scale
        pdf_y = pixel_top * scale  # Direct conversion - both from bottom
        
        pdf_width = pixel_width * scale
        pdf_height = pixel_height * scale
        
        try:
            data = dm.data.decode('utf-8').strip()
        except:
            data = str(dm.data)
        
        logger.info(f"DataMatrix '{data}': pixel_left={pixel_left}, pixel_top={pixel_top} (from bottom), image_height={image.height}")
        
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
    y_offset: float = -9
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
    return {"status": "healthy", "service": "lease-signer-api", "version": "1.3.0"}


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
    y_offset: float = Form(default=-9)
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


@app.post("/calibrate")
async def calibrate_signature_detection(
    pdf: UploadFile = File(...),
    page: Optional[int] = None
):
    """
    Calibration endpoint for signature detection.
    
    Analyzes a PDF and returns detailed signature detection results
    including pixel density values for threshold calibration.
    
    Args:
        pdf: The PDF file to analyze
        page: Optional - specific page number to analyze (1-indexed). 
              If not provided, analyzes all pages with TS_* codes.
    
    Returns:
        JSON with detection results for each tenant signature code found.
    """
    start_time = time.time()
    
    pdf_bytes = await pdf.read()
    
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)
        total_pages = len(images)
        
        results = {
            "filename": pdf.filename,
            "total_pages": total_pages,
            "pages_analyzed": [],
            "summary": {
                "signed": [],
                "unsigned": [],
                "uncertain": []
            },
            "density_stats": {
                "signed_densities": [],
                "unsigned_densities": []
            }
        }
        
        if page:
            if page < 1 or page > total_pages:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Page {page} out of range. PDF has {total_pages} pages."
                )
            pages_to_process = [page]
        else:
            pages_to_process = list(range(1, total_pages + 1))
        
        for page_num in pages_to_process:
            page_image = images[page_num - 1]
            
            is_phone_scan = detect_if_phone_scan(page_image)
            
            try:
                decoded = decode_datamatrix(page_image, timeout=10000)
            except Exception as e:
                results["pages_analyzed"].append({
                    "page": page_num,
                    "error": f"Decode error: {str(e)}"
                })
                continue
            
            if not decoded:
                continue
            
            detected_codes = []
            all_codes_found = []
            
            for code in decoded:
                try:
                    code_value = code.data.decode('utf-8')
                    rect = (code.rect.left, code.rect.top, code.rect.width, code.rect.height)
                    detected_codes.append((code_value, rect))
                    all_codes_found.append({
                        "code": code_value[:50] + "..." if len(code_value) > 50 else code_value,
                        "position": {"x": rect[0], "y": rect[1], "width": rect[2], "height": rect[3]}
                    })
                except Exception as e:
                    pass
            
            tenant_codes = [c for c, _ in detected_codes if c.startswith('TS_')]
            
            if not tenant_codes:
                continue
            
            sig_result = process_page_for_signatures(
                page_image,
                detected_codes,
                page_num
            )
            
            page_result = {
                "page": page_num,
                "is_phone_scan": is_phone_scan,
                "image_dimensions": {"width": page_image.width, "height": page_image.height},
                "all_codes_found": all_codes_found,
                "tenant_codes": sig_result["tenant_codes_found"],
                "all_signed": sig_result["all_signed"],
                "detection_details": sig_result["details"]
            }
            results["pages_analyzed"].append(page_result)
            
            for detail in sig_result["details"]:
                summary_item = {
                    "page": page_num,
                    "code": detail["code"],
                    "density": detail["pixel_density"],
                    "confidence": detail["confidence"]
                }
                
                if detail["status"] == "signed":
                    results["summary"]["signed"].append(summary_item)
                    results["density_stats"]["signed_densities"].append(detail["pixel_density"])
                elif detail["status"] == "unsigned":
                    results["summary"]["unsigned"].append(summary_item)
                    results["density_stats"]["unsigned_densities"].append(detail["pixel_density"])
                else:
                    results["summary"]["uncertain"].append(summary_item)
        
        signed_densities = results["density_stats"]["signed_densities"]
        unsigned_densities = results["density_stats"]["unsigned_densities"]
        
        results["density_stats"] = {
            "signed": {
                "count": len(signed_densities),
                "min": min(signed_densities) if signed_densities else None,
                "max": max(signed_densities) if signed_densities else None,
                "avg": sum(signed_densities) / len(signed_densities) if signed_densities else None
            },
            "unsigned": {
                "count": len(unsigned_densities),
                "min": min(unsigned_densities) if unsigned_densities else None,
                "max": max(unsigned_densities) if unsigned_densities else None,
                "avg": sum(unsigned_densities) / len(unsigned_densities) if unsigned_densities else None
            }
        }
        
        if signed_densities and unsigned_densities:
            max_unsigned = max(unsigned_densities)
            min_signed = min(signed_densities)
            
            if max_unsigned < min_signed:
                gap = min_signed - max_unsigned
                results["recommended_thresholds"] = {
                    "UNSIGNED_MAX_DENSITY": round(max_unsigned + (gap * 0.3), 4),
                    "SIGNED_MIN_DENSITY": round(min_signed - (gap * 0.3), 4),
                    "note": "Good separation between signed and unsigned"
                }
            else:
                results["recommended_thresholds"] = {
                    "warning": "Overlap between signed and unsigned densities",
                    "max_unsigned": max_unsigned,
                    "min_signed": min_signed,
                    "note": "May need to adjust detection region parameters"
                }
        
        results["processing_time_seconds"] = round(time.time() - start_time, 2)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calibrate/visual")
async def calibrate_signature_detection_visual(
    pdf: UploadFile = File(...),
    page: int = Form(default=1)
):
    """
    Visual calibration endpoint - returns an HTML page with annotated image
    showing detection regions.
    
    Args:
        pdf: The PDF file to analyze
        page: Page number to visualize (1-indexed, default=1)
    """
    pdf_bytes = await pdf.read()
    
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)
        total_pages = len(images)
        
        if page < 1 or page > total_pages:
            raise HTTPException(
                status_code=400,
                detail=f"Page {page} out of range. PDF has {total_pages} pages."
            )
        
        page_image = images[page - 1]
        
        decoded = decode_datamatrix(page_image, timeout=10000)
        
        if not decoded:
            raise HTTPException(
                status_code=400,
                detail=f"No DataMatrix codes found on page {page}"
            )
        
        detected_codes = []
        for code in decoded:
            try:
                code_value = code.data.decode('utf-8')
                rect = (code.rect.left, code.rect.top, code.rect.width, code.rect.height)
                detected_codes.append((code_value, rect))
            except:
                pass
        
        sig_result = process_page_for_signatures(
            page_image,
            detected_codes,
            page
        )
        
        annotated = page_image.copy()
        draw = ImageDraw.Draw(annotated)
        
        image_height = page_image.height
        
        for code_value, code_rect in detected_codes:
            code_x, code_y_from_bottom, code_width, code_height = code_rect
            code_y_from_top = image_height - code_y_from_bottom - code_height
            
            draw.rectangle(
                [code_x, code_y_from_top, code_x + code_width, code_y_from_top + code_height],
                outline="blue",
                width=2
            )
            
            label = code_value[:20] + "..." if len(code_value) > 20 else code_value
            draw.text((code_x, code_y_from_top - 15), label, fill="blue")
        
        for detail in sig_result["details"]:
            region = detail["region"]
            color = {
                "signed": "green",
                "unsigned": "red",
                "uncertain": "orange"
            }.get(detail["status"], "gray")
            
            draw.rectangle(
                [region[0], region[1], region[2], region[3]],
                outline=color,
                width=3
            )
            
            label = f"{detail['code']}: {detail['status']} ({detail['pixel_density']:.4f})"
            draw.text((region[0], region[1] - 15), label, fill=color)
        
        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Signature Detection Calibration - Page {page}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .image-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                img {{ max-width: 100%; height: auto; }}
                .legend {{ margin-top: 20px; padding: 15px; background: white; border-radius: 8px; }}
                .legend-item {{ display: inline-block; margin-right: 20px; padding: 5px 10px; }}
                .results {{ margin-top: 20px; padding: 15px; background: white; border-radius: 8px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f0f0f0; }}
                .signed {{ color: green; font-weight: bold; }}
                .unsigned {{ color: red; font-weight: bold; }}
                .uncertain {{ color: orange; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Signature Detection Calibration</h1>
                <p>File: {pdf.filename} | Page: {page} of {total_pages}</p>
                
                <div class="legend">
                    <strong>Legend:</strong>
                    <span class="legend-item" style="border-left: 4px solid blue; padding-left: 8px;">Blue = DataMatrix code</span>
                    <span class="legend-item" style="border-left: 4px solid green; padding-left: 8px;">Green = Signed</span>
                    <span class="legend-item" style="border-left: 4px solid red; padding-left: 8px;">Red = Unsigned</span>
                    <span class="legend-item" style="border-left: 4px solid orange; padding-left: 8px;">Orange = Uncertain</span>
                </div>
                
                <div class="image-container">
                    <img src="data:image/png;base64,{img_base64}" alt="Annotated page">
                </div>
                
                <div class="results">
                    <h2>Detection Results</h2>
                    <p><strong>Phone scan detected:</strong> {sig_result.get('is_phone_scan', False)}</p>
                    <p><strong>Tenant codes found:</strong> {', '.join(sig_result['tenant_codes_found']) or 'None'}</p>
                    
                    <h3>Details</h3>
                    <table>
                        <tr>
                            <th>Code</th>
                            <th>Status</th>
                            <th>Confidence</th>
                            <th>Pixel Density</th>
                            <th>Detection Region</th>
                        </tr>
                        {''.join(f"""
                        <tr>
                            <td>{d['code']}</td>
                            <td class="{d['status']}">{d['status'].upper()}</td>
                            <td>{d['confidence']:.1%}</td>
                            <td>{d['pixel_density']:.4f}</td>
                            <td>x:{d['region'][0]}, y:{d['region'][1]}, w:{d['region'][2]-d['region'][0]}, h:{d['region'][3]-d['region'][1]}</td>
                        </tr>
                        """ for d in sig_result['details'])}
                    </table>
                </div>
                
                <div class="results">
                    <h2>All Codes Found on Page</h2>
                    <ul>
                        {''.join(f'<li>{c[:60]}{"..." if len(c) > 60 else ""}</li>' for c, _ in detected_codes)}
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug")
async def debug_visual(
    pdf: UploadFile = File(...),
    dpi: int = Form(default=150),
    page: int = Form(default=0)  # 0 = all pages, or specific page number
):
    """
    Visual debug endpoint - returns HTML page showing detected DataMatrix codes
    with hover info displaying exact coordinates.
    """
    pdf_bytes = await pdf.read()
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(pdf_reader.pages)
    scale = 72.0 / dpi
    
    # Determine which pages to process
    if page > 0:
        pages_to_process = [page]
    else:
        pages_to_process = list(range(1, total_pages + 1))
    
    pages_html = []
    
    for page_num in pages_to_process:
        if page_num > total_pages:
            continue
            
        # Get page dimensions
        pdf_page = pdf_reader.pages[page_num - 1]
        page_width = float(pdf_page.mediabox.width)
        page_height = float(pdf_page.mediabox.height)
        
        # Convert to image
        images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=page_num, last_page=page_num)
        if not images:
            continue
        img = images[0]
        
        # Detect DataMatrix codes
        decoded = decode_datamatrix(img)
        
        overlays = []
        code_names = []
        
        for dm in decoded:
            try:
                code = dm.data.decode('utf-8').strip()
            except:
                code = str(dm.data)
            
            code_names.append(code)
            is_signature = code in ALL_SIGNATURE_CODES
            code_type = "signature" if is_signature else "metadata"
            
            pixel_left = dm.rect.left
            pixel_top = dm.rect.top
            pixel_width = dm.rect.width
            pixel_height = dm.rect.height
            
            data_info = {
                'code': code,
                'pixel_left': pixel_left,
                'pixel_top': pixel_top,
                'pixel_width': pixel_width,
                'pixel_height': pixel_height,
                'image_height': img.height,
                'image_width': img.width,
                'page_height': page_height,
                'page_width': page_width,
                'scale': scale,
            }
            
            data_json = json.dumps(data_info)
            overlay = f'''<div class="code-overlay {code_type}" 
                 style="left:{pixel_left}px; top:{pixel_top}px; width:{pixel_width}px; height:{pixel_height}px;"
                 data-info='{data_json}'
                 title="{code}"></div>'''
            overlays.append(overlay)
        
        # Convert image to base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        page_html = f'''
        <div class="page-container">
            <div class="page-title">Page {page_num} - {len(decoded)} codes found</div>
            <div class="coords">
                Image: {img.width}x{img.height} px | Page: {page_width:.1f}x{page_height:.1f} pts | Scale: {scale:.4f}
            </div>
            <div class="image-wrapper" style="width:{img.width}px; height:{img.height}px;">
                <img src="data:image/png;base64,{img_base64}" width="{img.width}" height="{img.height}">
                {''.join(overlays)}
            </div>
            <div class="coords">
                Codes found: {', '.join(code_names) if code_names else 'None'}
            </div>
        </div>
        '''
        pages_html.append(page_html)
        del img
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DataMatrix Debug Viewer</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }}
            h1 {{ color: #4CAF50; }}
            .page-container {{ margin: 20px 0; background: #2d2d2d; padding: 20px; border-radius: 8px; }}
            .page-title {{ font-size: 18px; margin-bottom: 10px; color: #81C784; }}
            .image-wrapper {{ position: relative; display: inline-block; }}
            .image-wrapper img {{ max-width: 100%; border: 1px solid #444; }}
            .code-overlay {{
                position: absolute;
                border: 3px solid;
                background: rgba(255,255,255,0.1);
                cursor: pointer;
                box-sizing: border-box;
            }}
            .code-overlay.signature {{ border-color: #4CAF50; }}
            .code-overlay.metadata {{ border-color: #2196F3; }}
            .code-overlay:hover {{ background: rgba(255,255,255,0.3); }}
            .tooltip {{
                position: absolute;
                background: rgba(0,0,0,0.95);
                color: #fff;
                padding: 12px;
                border-radius: 4px;
                font-size: 13px;
                z-index: 1000;
                pointer-events: none;
                white-space: pre;
                border: 1px solid #4CAF50;
                font-family: monospace;
            }}
            .legend {{ margin: 20px 0; padding: 15px; background: #2d2d2d; border-radius: 8px; }}
            .legend span {{ margin-right: 20px; }}
            .legend .sig {{ color: #4CAF50; }}
            .legend .meta {{ color: #2196F3; }}
            .info {{ background: #333; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            .coords {{ font-family: monospace; background: #1a1a1a; padding: 10px; margin: 10px 0; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>DataMatrix Debug Viewer</h1>
        <div class="info">
            <strong>DPI:</strong> {dpi} | <strong>Scale (72/DPI):</strong> {scale:.4f} | <strong>Total Pages:</strong> {total_pages}
        </div>
        <div class="legend">
            <span class="sig">■ Signature Codes (green)</span>
            <span class="meta">■ Metadata Codes (blue)</span>
        </div>
        
        {''.join(pages_html)}
        
        <div id="tooltip" class="tooltip" style="display:none;"></div>
        
        <script>
            const tooltip = document.getElementById('tooltip');
            
            document.querySelectorAll('.code-overlay').forEach(el => {{
                el.addEventListener('mouseenter', (e) => {{
                    const d = JSON.parse(el.dataset.info);
                    const pctFromTop = (d.pixel_top / d.image_height * 100).toFixed(1);
                    const pctFromBottom = ((d.image_height - d.pixel_top - d.pixel_height) / d.image_height * 100).toFixed(1);
                    const pdfY_topOrigin = (d.page_height - (d.pixel_top + d.pixel_height) * d.scale).toFixed(1);
                    const pdfY_bottomOrigin = (d.pixel_top * d.scale).toFixed(1);
                    
                    tooltip.innerHTML = `<strong>${{d.code}}</strong>

PIXEL COORDINATES (from pylibdmtx):
  rect.left:   ${{d.pixel_left}} px
  rect.top:    ${{d.pixel_top}} px
  rect.width:  ${{d.pixel_width}} px
  rect.height: ${{d.pixel_height}} px

IMAGE INFO:
  Size: ${{d.image_width}} x ${{d.image_height}} px
  Code is ${{pctFromTop}}% from TOP
  Code is ${{pctFromBottom}}% from BOTTOM

PDF Y-COORDINATE CALCULATIONS:
  If rect.top = from IMAGE TOP:
    pdf_y = ${{pdfY_topOrigin}} pts (from page bottom)
    
  If rect.top = from IMAGE BOTTOM:
    pdf_y = ${{pdfY_bottomOrigin}} pts (from page bottom)`;
                    tooltip.style.display = 'block';
                }});
                
                el.addEventListener('mousemove', (e) => {{
                    tooltip.style.left = (e.pageX + 15) + 'px';
                    tooltip.style.top = (e.pageY + 15) + 'px';
                }});
                
                el.addEventListener('mouseleave', () => {{
                    tooltip.style.display = 'none';
                }});
            }});
        </script>
    </body>
    </html>
    '''
    
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
