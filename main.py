"""
Lease Signing Microservice
Detects DataMatrix codes in PDFs and places signature images at marked locations.

DataMatrix codes expected:
- TS_MAIN, TS_RIDER_A, TS_RIDER_B, TS_RIDER_C, TS_BEDBUG, TS_WIND (tenant signatures)
- LS_MAIN, LS_RIDER_A, LS_RIDER_B, LS_RIDER_C, LS_BEDBUG (landlord signatures)
"""

import io
import os
import tempfile
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lease Signer API",
    description="Detects DataMatrix signature markers in PDFs and applies signatures",
    version="1.0.0"
)

# Signature code mappings
TENANT_SIGNATURE_CODES = {
    "TS_MAIN", "TS_RIDER_A", "TS_RIDER_B", "TS_RIDER_C", "TS_BEDBUG", "TS_WIND"
}
LANDLORD_SIGNATURE_CODES = {
    "LS_MAIN", "LS_RIDER_A", "LS_RIDER_B", "LS_RIDER_C", "LS_BEDBUG"
}
ALL_SIGNATURE_CODES = TENANT_SIGNATURE_CODES | LANDLORD_SIGNATURE_CODES


def find_datamatrix_codes(image: Image.Image, dpi: int = 300) -> list[dict]:
    """
    Find and decode all DataMatrix codes in an image.
    Returns list of dicts with 'data', 'x', 'y', 'width', 'height' in PDF points.
    """
    # Convert PIL Image to format pylibdmtx expects
    decoded = decode_datamatrix(image)
    
    results = []
    for dm in decoded:
        # dm.rect is (left, top, width, height) in pixels
        # Convert pixels to PDF points (72 points per inch)
        scale = 72.0 / dpi
        
        # pylibdmtx returns rect as (left, top, width, height)
        # but 'top' is from the top of the image in pixel coords
        # PDF coords are from bottom-left, so we need to flip Y
        img_height = image.height
        
        pixel_left = dm.rect.left
        pixel_top = dm.rect.top
        pixel_width = dm.rect.width
        pixel_height = dm.rect.height
        
        # Convert to PDF coordinates (bottom-left origin)
        pdf_x = pixel_left * scale
        pdf_y = (img_height - pixel_top - pixel_height) * scale
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
            'pixel_rect': {
                'left': pixel_left,
                'top': pixel_top,
                'width': pixel_width,
                'height': pixel_height
            }
        })
        logger.info(f"Found DataMatrix: '{data}' at PDF coords ({pdf_x:.1f}, {pdf_y:.1f})")
    
    return results


def create_signature_overlay(
    page_width: float,
    page_height: float,
    signatures_to_place: list[dict],
    tenant_sig_image: Optional[bytes],
    landlord_sig_image: Optional[bytes],
    sig_width: float = 100,
    sig_height: float = 40,
    x_offset: float = 0,
    y_offset: float = -5
) -> bytes:
    """
    Create a PDF page with signature images placed at specified locations.
    Returns PDF bytes for a single page overlay.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    
    for sig_info in signatures_to_place:
        code = sig_info['data']
        
        # Determine which signature image to use
        if code in TENANT_SIGNATURE_CODES and tenant_sig_image:
            sig_bytes = tenant_sig_image
        elif code in LANDLORD_SIGNATURE_CODES and landlord_sig_image:
            sig_bytes = landlord_sig_image
        else:
            logger.warning(f"No signature image for code: {code}")
            continue
        
        # Place signature relative to DataMatrix position
        # Position signature to the right and slightly below the DataMatrix
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
    """Merge an overlay PDF onto an original page (modifies original_page in place)."""
    overlay_reader = PdfReader(io.BytesIO(overlay_pdf_bytes))
    if len(overlay_reader.pages) > 0:
        original_page.merge_page(overlay_reader.pages[0])


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "lease-signer-api"}


@app.post("/scan")
async def scan_pdf_for_codes(
    pdf_file: UploadFile = File(...),
    dpi: int = Form(default=200)
):
    """
    Scan a PDF and return all DataMatrix codes found with their locations.
    Useful for debugging/testing before applying signatures.
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await pdf_file.read()
    
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
    except Exception as e:
        raise HTTPException(500, f"Failed to convert PDF to images: {e}")
    
    all_codes = []
    for page_num, image in enumerate(images, start=1):
        codes = find_datamatrix_codes(image, dpi=dpi)
        for code in codes:
            code['page'] = page_num
            all_codes.append(code)
    
    # Separate signature codes from metadata codes
    signature_codes = [c for c in all_codes if c['data'] in ALL_SIGNATURE_CODES]
    metadata_codes = [c for c in all_codes if c['data'] not in ALL_SIGNATURE_CODES]
    
    return {
        "total_pages": len(images),
        "total_codes_found": len(all_codes),
        "signature_markers": signature_codes,
        "metadata_codes": metadata_codes
    }


@app.post("/sign")
async def sign_pdf(
    pdf_file: UploadFile = File(...),
    tenant_signature: Optional[UploadFile] = File(default=None),
    landlord_signature: Optional[UploadFile] = File(default=None),
    dpi: int = Form(default=200),
    sig_width: float = Form(default=100),
    sig_height: float = Form(default=40),
    x_offset: float = Form(default=0),
    y_offset: float = Form(default=-5)
):
    """
    Sign a PDF by detecting DataMatrix markers and placing signatures.
    
    - pdf_file: The lease PDF with DataMatrix codes
    - tenant_signature: PNG image for tenant signature (applied to TS_* codes)
    - landlord_signature: PNG image for landlord signature (applied to LS_* codes)
    - dpi: Resolution for scanning (higher = more accurate but slower)
    - sig_width: Width of signature in PDF points (72 points = 1 inch)
    - sig_height: Height of signature in PDF points
    - x_offset: Horizontal offset from DataMatrix position
    - y_offset: Vertical offset from DataMatrix position
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
    
    # Convert PDF pages to images for scanning
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
    except Exception as e:
        raise HTTPException(500, f"Failed to convert PDF to images: {e}")
    
    # Read original PDF
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    pdf_writer = PdfWriter()
    
    signatures_applied = []
    
    for page_num, (image, original_page) in enumerate(zip(images, pdf_reader.pages), start=1):
        # Get page dimensions
        page_width = float(original_page.mediabox.width)
        page_height = float(original_page.mediabox.height)
        
        # Find DataMatrix codes on this page
        codes = find_datamatrix_codes(image, dpi=dpi)
        
        # Filter to only signature codes
        sig_codes = [c for c in codes if c['data'] in ALL_SIGNATURE_CODES]
        
        if sig_codes:
            logger.info(f"Page {page_num}: Found {len(sig_codes)} signature markers")
            
            # Create overlay with signatures
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
            
            # Merge overlay onto original page
            merge_overlay_with_page(original_page, overlay_bytes)
            
            for sc in sig_codes:
                signatures_applied.append({
                    'page': page_num,
                    'code': sc['data'],
                    'x': sc['x'],
                    'y': sc['y']
                })
        
        pdf_writer.add_page(original_page)
    
    # Write output PDF
    output_buffer = io.BytesIO()
    pdf_writer.write(output_buffer)
    output_buffer.seek(0)
    
    logger.info(f"Signing complete. Applied {len(signatures_applied)} signatures.")
    
    # Return the signed PDF
    return Response(
        content=output_buffer.read(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=signed_{pdf_file.filename}",
            "X-Signatures-Applied": str(len(signatures_applied)),
            "X-Signature-Details": str(signatures_applied)
        }
    )


@app.post("/sign-with-details")
async def sign_pdf_with_details(
    pdf_file: UploadFile = File(...),
    tenant_signature: Optional[UploadFile] = File(default=None),
    landlord_signature: Optional[UploadFile] = File(default=None),
    dpi: int = Form(default=200),
    sig_width: float = Form(default=100),
    sig_height: float = Form(default=40),
    x_offset: float = Form(default=0),
    y_offset: float = Form(default=-5)
):
    """
    Same as /sign but returns JSON with base64 PDF and detailed results.
    Useful when you need metadata about what was signed.
    """
    import base64
    
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
    
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
    except Exception as e:
        raise HTTPException(500, f"Failed to convert PDF to images: {e}")
    
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    pdf_writer = PdfWriter()
    
    signatures_applied = []
    all_codes_found = []
    
    for page_num, (image, original_page) in enumerate(zip(images, pdf_reader.pages), start=1):
        page_width = float(original_page.mediabox.width)
        page_height = float(original_page.mediabox.height)
        
        codes = find_datamatrix_codes(image, dpi=dpi)
        
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
    
    return {
        "success": True,
        "total_pages": len(images),
        "signatures_applied": signatures_applied,
        "signatures_count": len(signatures_applied),
        "all_codes_found": [
            {"page": c['page'], "data": c['data'], "x": round(c['x'], 2), "y": round(c['y'], 2)}
            for c in all_codes_found
        ],
        "pdf_base64": pdf_base64
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
