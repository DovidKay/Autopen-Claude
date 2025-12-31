"""
Lease Signer API - v1.5.0

FastAPI microservice for processing lease PDFs:
- Detects DataMatrix codes
- Extracts metadata for logging
- Places signature images at designated locations
- Returns structured response with validation info

New in v1.5.0:
- Swagger UI Authorize button for API key
- Improved page validation tracking
"""

import os
import io
import uuid
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
import fitz  # PyMuPDF
from pylibdmtx.pylibdmtx import decode as dmtx_decode

# API Key authentication
API_KEY = os.environ.get("API_KEY", "")  # Set this in Render environment variables
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key from header"""
    if not API_KEY:
        return True  # No API key configured, allow all requests
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

app = FastAPI(
    title="Lease Signer API", 
    version="1.5.0",
    description="API for signing lease PDFs with DataMatrix code detection"
)

# Job storage
jobs: Dict[str, dict] = {}

# Detection parameters
DPI = 150
SCALE = DPI / 72  # PDF points to pixels


@dataclass
class LeaseMetadata:
    """Parsed metadata from DataMatrix code"""
    building: str = ""
    unit: str = ""
    tenant: str = ""
    lease_date: str = ""
    user_id: str = ""
    submission_id: str = ""
    current_page: int = 0
    total_pages: int = 0
    raw: str = ""
    
    def to_dict(self) -> dict:
        return {
            "building": self.building,
            "unit": self.unit,
            "tenant": self.tenant,
            "lease_date": self.lease_date,
            "user_id": self.user_id,
            "submission_id": self.submission_id,
            "total_pages": self.total_pages
        }


@dataclass
class ValidationResult:
    """Validation results for the lease PDF"""
    pages_found: int = 0
    pages_expected: int = 0
    pages_with_metadata: List[int] = field(default_factory=list)
    missing_pages: List[int] = field(default_factory=list)
    ls_codes_found: List[str] = field(default_factory=list)
    ts_codes_found: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    page_details: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "pages_found": self.pages_found,
            "pages_expected": self.pages_expected,
            "pages_with_metadata": self.pages_with_metadata,
            "missing_pages": self.missing_pages,
            "ls_codes_found": self.ls_codes_found,
            "ts_codes_found": self.ts_codes_found,
            "warnings": self.warnings,
            "page_details": self.page_details
        }


def parse_metadata(code_value: str) -> Optional[LeaseMetadata]:
    """
    Parse metadata from DataMatrix code.
    Format: '2001-|1B|David Kornitzer612|12/01/2025|1111|6428507983527856062|Pg:1/12'
    """
    try:
        parts = code_value.split('|')
        if len(parts) < 7:
            return None
        
        # Parse page info (e.g., "Pg:1/12")
        page_info = parts[6] if len(parts) > 6 else ""
        current_page = 0
        total_pages = 0
        if page_info.startswith("Pg:"):
            page_parts = page_info[3:].split('/')
            if len(page_parts) == 2:
                current_page = int(page_parts[0])
                total_pages = int(page_parts[1])
        
        return LeaseMetadata(
            building=parts[0],
            unit=parts[1],
            tenant=parts[2],
            lease_date=parts[3],
            user_id=parts[4],
            submission_id=parts[5],
            current_page=current_page,
            total_pages=total_pages,
            raw=code_value
        )
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return None


def generate_output_filename(metadata: LeaseMetadata) -> str:
    """
    Generate output filename from metadata.
    Format: 'Lease {building}{unit} {tenant} {date} -signed.pdf'
    Example: 'Lease 2001-1B David Kornitzer612 12-1-2025 -signed.pdf'
    """
    if not metadata or not metadata.building:
        return f"lease-signed-{uuid.uuid4().hex[:8]}.pdf"
    
    # Format date (convert from MM/DD/YYYY to M-D-YYYY)
    date_formatted = metadata.lease_date
    try:
        date_obj = datetime.strptime(metadata.lease_date, "%m/%d/%Y")
        date_formatted = f"{date_obj.month}-{date_obj.day}-{date_obj.year}"
    except:
        date_formatted = metadata.lease_date.replace('/', '-')
    
    # Clean tenant name (remove any characters that might cause issues)
    tenant_clean = metadata.tenant.replace('/', '-').replace('\\', '-')
    
    filename = f"Lease {metadata.building}{metadata.unit} {tenant_clean} {date_formatted} -signed.pdf"
    return filename


def detect_codes_on_page(page_image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Detect all DataMatrix codes on a page image."""
    detected = dmtx_decode(page_image, timeout=10000)
    results = []
    for d in detected:
        code_value = d.data.decode('utf-8')
        rect = d.rect
        # rect: (left, top, width, height) - top is from BOTTOM of image
        results.append((code_value, (rect.left, rect.top, rect.width, rect.height)))
    return results


def is_signature_code(code: str) -> bool:
    """Check if code is a signature marker (LS_* or TS_*)"""
    return code.startswith('LS_') or code.startswith('TS_')


def is_metadata_code(code: str) -> bool:
    """Check if code contains metadata (has pipe separators)"""
    return '|' in code


def process_pdf(
    pdf_bytes: bytes,
    landlord_sig_bytes: bytes,
    tenant_sig_bytes: Optional[bytes] = None
) -> Tuple[bytes, LeaseMetadata, ValidationResult]:
    """
    Process the lease PDF:
    1. Detect all DataMatrix codes
    2. Extract metadata
    3. Place signatures
    4. Return signed PDF with metadata and validation
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Load signature images - create fresh BytesIO for each
    landlord_sig_io = io.BytesIO(landlord_sig_bytes)
    landlord_sig = Image.open(landlord_sig_io)
    landlord_sig.load()  # Force load the image data
    
    tenant_sig = None
    if tenant_sig_bytes:
        tenant_sig_io = io.BytesIO(tenant_sig_bytes)
        tenant_sig = Image.open(tenant_sig_io)
        tenant_sig.load()  # Force load the image data
    
    # Tracking
    metadata: Optional[LeaseMetadata] = None
    validation = ValidationResult()
    validation.pages_found = len(doc)
    all_codes_found = []
    pages_with_metadata_codes = []
    page_details = []  # Track what was found on each page
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_info = {"page": page_num + 1, "metadata_found": False, "signature_codes": [], "metadata_page_number": None}
        
        # Render page to image
        mat = fitz.Matrix(SCALE, SCALE)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Detect codes
        codes = detect_codes_on_page(img)
        
        for code_value, rect in codes:
            all_codes_found.append(code_value)
            
            # Extract metadata from metadata codes
            if is_metadata_code(code_value):
                page_info["metadata_found"] = True
                pages_with_metadata_codes.append(page_num + 1)
                
                # Parse to get the page number from the metadata itself
                parsed = parse_metadata(code_value)
                if parsed:
                    page_info["metadata_page_number"] = parsed.current_page
                    
                    # Use first complete metadata for the document
                    if metadata is None:
                        metadata = parsed
                        validation.pages_expected = metadata.total_pages
            
            # Track signature codes
            if code_value.startswith('LS_'):
                page_info["signature_codes"].append(code_value)
                if code_value not in validation.ls_codes_found:
                    validation.ls_codes_found.append(code_value)
            elif code_value.startswith('TS_'):
                page_info["signature_codes"].append(code_value)
                if code_value not in validation.ts_codes_found:
                    validation.ts_codes_found.append(code_value)
            
            # Place landlord signature at LS_* codes
            if code_value.startswith('LS_'):
                place_signature(doc, page, rect, landlord_sig, pix.height)
            
            # Place tenant signature at TS_* codes (if provided)
            if code_value.startswith('TS_') and tenant_sig:
                place_signature(doc, page, rect, tenant_sig, pix.height)
        
        page_details.append(page_info)
    
    # Validation checks
    validation.pages_with_metadata = pages_with_metadata_codes
    validation.page_details = page_details
    
    # Check for missing pages (pages without metadata codes)
    if validation.pages_expected > 0:
        expected_pages = set(range(1, validation.pages_expected + 1))
        found_pages = set(pages_with_metadata_codes)
        validation.missing_pages = sorted(list(expected_pages - found_pages))
        
        if validation.missing_pages:
            validation.warnings.append(f"Missing metadata on pages: {validation.missing_pages}")
    
    # Check page count mismatch
    if validation.pages_expected > 0 and validation.pages_found != validation.pages_expected:
        validation.warnings.append(
            f"Page count mismatch: found {validation.pages_found}, expected {validation.pages_expected}"
        )
    
    # Check for missing signature codes
    if not validation.ls_codes_found:
        validation.warnings.append("No landlord signature codes (LS_*) found")
    if not validation.ts_codes_found:
        validation.warnings.append("No tenant signature codes (TS_*) found")
    
    # Save to bytes
    output = io.BytesIO()
    doc.save(output)
    doc.close()
    
    return output.getvalue(), metadata or LeaseMetadata(), validation


def place_signature(doc, page, rect: Tuple[int, int, int, int], sig_image: Image.Image, img_height: int):
    """
    Place signature image on the page.
    
    rect: (left, top, width, height) from pylibdmtx
    - left: x from left edge of image
    - top: y from BOTTOM of image (pylibdmtx uses bottom-left origin)
    - width, height: dimensions of the detected code
    
    PyMuPDF (fitz) uses TOP-LEFT origin for page coordinates.
    We need to convert Y coordinate: fitz_y = page_height - pylibdmtx_y
    """
    left, top, width, height = rect
    
    # Get page dimensions in points
    page_height = page.rect.height
    
    # Convert pixel coordinates to PDF points
    pdf_x = left / SCALE
    code_y_from_bottom = top / SCALE  # pylibdmtx: distance from bottom
    code_width = width / SCALE
    code_height = height / SCALE
    
    # Convert Y from bottom-left origin to top-left origin
    # The code's TOP edge in top-left coordinates
    pdf_y = page_height - code_y_from_bottom - code_height
    
    # Signature placement: to the RIGHT of the code, vertically centered
    sig_width = 100  # points
    sig_height = 40  # points
    
    sig_x = pdf_x + code_width + 5  # 5 points gap to the right
    sig_y = pdf_y + (code_height / 2) - (sig_height / 2)  # Vertically centered with code
    
    # Create signature rect (x0, y0, x1, y1) - top-left to bottom-right
    sig_rect = fitz.Rect(sig_x, sig_y, sig_x + sig_width, sig_y + sig_height)
    
    # Convert PIL image to bytes
    sig_bytes = io.BytesIO()
    sig_image.save(sig_bytes, format='PNG')
    sig_bytes.seek(0)
    
    # Insert image
    page.insert_image(sig_rect, stream=sig_bytes.getvalue())


def process_job(job_id: str, pdf_bytes: bytes, landlord_sig_bytes: bytes, tenant_sig_bytes: Optional[bytes]):
    """Background job processor"""
    import traceback
    try:
        jobs[job_id]["status"] = "processing"
        print(f"[{job_id}] Starting PDF processing...")
        
        # Process the PDF
        signed_pdf, metadata, validation = process_pdf(pdf_bytes, landlord_sig_bytes, tenant_sig_bytes)
        
        print(f"[{job_id}] PDF processing complete, generating filename...")
        
        # Generate filename
        output_filename = generate_output_filename(metadata)
        
        print(f"[{job_id}] Output filename: {output_filename}")
        
        # Store results
        jobs[job_id].update({
            "status": "complete",
            "signed_pdf": signed_pdf,
            "output_filename": output_filename,
            "metadata": metadata.to_dict(),
            "validation": validation.to_dict(),
            "signatures_applied": len(validation.ls_codes_found) + (len(validation.ts_codes_found) if tenant_sig_bytes else 0),
            "completed_at": datetime.utcnow().isoformat()
        })
        
        print(f"[{job_id}] Job complete!")
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{job_id}] ERROR: {error_msg}")
        jobs[job_id].update({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "completed_at": datetime.utcnow().isoformat()
        })


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint for Render health check"""
    return {"status": "healthy", "version": "1.5.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.5.0"}


@app.post("/sign")
async def sign_lease(
    pdf: UploadFile = File(...),
    landlord_signature: UploadFile = File(...),
    tenant_signature: Optional[UploadFile] = File(None),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Submit a lease PDF for signing.
    Returns a job_id for status polling.
    """
    job_id = uuid.uuid4().hex
    
    # Read files
    pdf_bytes = await pdf.read()
    landlord_sig_bytes = await landlord_signature.read()
    tenant_sig_bytes = await tenant_signature.read() if tenant_signature else None
    
    # Validate files
    if not pdf_bytes or len(pdf_bytes) < 100:
        raise HTTPException(status_code=400, detail=f"PDF file is empty or too small ({len(pdf_bytes) if pdf_bytes else 0} bytes)")
    
    if not landlord_sig_bytes or len(landlord_sig_bytes) < 100:
        raise HTTPException(status_code=400, detail=f"Landlord signature file is empty or too small ({len(landlord_sig_bytes) if landlord_sig_bytes else 0} bytes)")
    
    # Validate signature is a valid image
    try:
        test_img = Image.open(io.BytesIO(landlord_sig_bytes))
        test_img.verify()  # Verify it's a valid image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Landlord signature is not a valid image: {str(e)}")
    
    # Initialize job
    jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "original_filename": pdf.filename
    }
    
    # Start background processing
    thread = threading.Thread(
        target=process_job,
        args=(job_id, pdf_bytes, landlord_sig_bytes, tenant_sig_bytes)
    )
    thread.start()
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Processing started"
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Check job status.
    Returns metadata and validation info when complete.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"]
    }
    
    if job["status"] == "complete":
        response.update({
            "output_filename": job.get("output_filename"),
            "metadata": job.get("metadata"),
            "validation": job.get("validation"),
            "signatures_applied": job.get("signatures_applied"),
            "download_url": f"/download/{job_id}"
        })
    elif job["status"] == "error":
        response["error"] = job.get("error")
    
    return response


@app.get("/download/{job_id}")
async def download_signed_pdf(job_id: str):
    """Download the signed PDF with dynamic filename"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    
    if "signed_pdf" not in job:
        raise HTTPException(status_code=500, detail="Signed PDF not available")
    
    # Get the dynamic filename
    filename = job.get("output_filename", f"lease-signed-{job_id}.pdf")
    
    # Save to temp file for FileResponse
    temp_path = f"/tmp/{job_id}.pdf"
    with open(temp_path, "wb") as f:
        f.write(job["signed_pdf"])
    
    return FileResponse(
        temp_path,
        media_type="application/pdf",
        filename=filename
    )


@app.post("/calibrate")
async def calibrate_detection(
    pdf: UploadFile = File(...),
    page: int = Form(1)
):
    """
    Calibration endpoint for signature detection testing.
    Analyzes a specific page and returns detection data.
    """
    from signature_detection import process_page_for_signatures, detect_if_phone_scan
    
    pdf_bytes = await pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if page < 1 or page > len(doc):
        raise HTTPException(status_code=400, detail=f"Invalid page number. PDF has {len(doc)} pages.")
    
    start_time = time.time()
    
    # Render page
    target_page = doc[page - 1]
    mat = fitz.Matrix(SCALE, SCALE)
    pix = target_page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Detect codes
    codes = detect_codes_on_page(img)
    
    # Process for signatures
    detected_codes = [(code, rect) for code, rect in codes]
    detection_result = process_page_for_signatures(img, detected_codes, page)
    
    # Extract metadata
    metadata = None
    for code, rect in codes:
        if is_metadata_code(code):
            metadata = parse_metadata(code)
            break
    
    processing_time = time.time() - start_time
    doc.close()
    
    return {
        "filename": pdf.filename,
        "total_pages": len(doc) if doc else 0,
        "pages_analyzed": [{
            "page": page,
            "is_phone_scan": detection_result.get("is_phone_scan"),
            "image_dimensions": {"width": pix.width, "height": pix.height},
            "all_codes_found": [
                {"code": code, "position": {"x": r[0], "y": r[1], "width": r[2], "height": r[3]}}
                for code, r in codes
            ],
            "tenant_codes": detection_result.get("tenant_codes_found", []),
            "all_signed": detection_result.get("all_signed"),
            "detection_details": detection_result.get("details", [])
        }],
        "summary": {
            "signed": [d for d in detection_result.get("details", []) if d["status"] == "signed"],
            "unsigned": [d for d in detection_result.get("details", []) if d["status"] == "unsigned"],
            "uncertain": [d for d in detection_result.get("details", []) if d["status"] == "uncertain"]
        },
        "metadata": metadata.to_dict() if metadata else None,
        "processing_time_seconds": round(processing_time, 2)
    }


@app.post("/debug")
async def debug_detection(
    pdf: UploadFile = File(...),
    page: int = Form(1)
):
    """
    Debug endpoint - returns HTML visualization of detected codes.
    """
    import base64
    
    pdf_bytes = await pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if page < 1 or page > len(doc):
        raise HTTPException(status_code=400, detail=f"Invalid page number. PDF has {len(doc)} pages.")
    
    # Render page
    target_page = doc[page - 1]
    mat = fitz.Matrix(SCALE, SCALE)
    pix = target_page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Detect codes
    codes = detect_codes_on_page(img)
    
    # Convert image to base64
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    # Build HTML
    code_boxes = ""
    code_list = ""
    for code, rect in codes:
        left, top, width, height = rect
        # Convert top from bottom-origin to top-origin for CSS
        css_top = pix.height - top - height
        code_boxes += f'''
        <div class="code-box" style="left:{left}px; top:{css_top}px; width:{width}px; height:{height}px;"
             title="{code}"></div>
        '''
        code_list += f"<li><strong>{code}</strong> - x:{left}, y:{top}, w:{width}, h:{height}</li>"
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .container {{ position: relative; display: inline-block; }}
            .code-box {{ position: absolute; border: 2px solid blue; background: rgba(0,0,255,0.1); }}
            img {{ max-width: 100%; }}
            .results {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h2>Page {page} - {len(codes)} codes detected</h2>
        <div class="container">
            <img src="data:image/png;base64,{img_base64}">
            {code_boxes}
        </div>
        <div class="results">
            <h3>Codes Found:</h3>
            <ul>{code_list}</ul>
        </div>
    </body>
    </html>
    '''
    
    doc.close()
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
