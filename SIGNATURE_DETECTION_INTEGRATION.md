# Signature Detection Integration Guide

## Overview

This guide shows how to integrate the tenant signature detection feature into your existing `lease-signer-api`.

## New Workflow Logic

```
1. Receive PDF
2. For each page with TS_* codes:
   a. Render page to image
   b. Detect DataMatrix codes
   c. Check if tenant has signed (handwritten mark present)
3. Decision:
   - If ALL tenant signatures detected → Apply landlord signatures → Return signed PDF
   - If ANY tenant signatures missing → Flag document → Return error with details
```

## Files to Add/Modify

### 1. Add `signature_detection.py` (new file)

Copy the `signature_detection.py` module to your repository.

### 2. Modify `main.py`

#### Add imports:

```python
from signature_detection import process_page_for_signatures, SignatureStatus
```

#### Modify the signing logic:

```python
# In your existing sign endpoint / background task

async def process_signing(job_id: str, pdf_bytes: bytes, landlord_sig: bytes, tenant_sig: bytes):
    """Modified signing process with tenant signature detection."""
    
    try:
        # Render PDF pages to images (you already do this for DataMatrix detection)
        images = convert_from_bytes(pdf_bytes, dpi=150)
        
        all_signature_results = []
        unsigned_pages = []
        
        for page_num, page_image in enumerate(images, 1):
            # Detect DataMatrix codes (your existing code)
            decoded = decode(page_image, timeout=10000)
            detected_codes = [
                (code.data.decode('utf-8'), 
                 (code.rect.left, code.rect.top, code.rect.width, code.rect.height))
                for code in decoded
            ]
            
            # Check for tenant signature codes on this page
            tenant_codes = [c for c, _ in detected_codes if c.startswith('TS_')]
            
            if tenant_codes:
                # Run signature detection
                sig_result = process_page_for_signatures(
                    page_image,
                    detected_codes,
                    page_num
                )
                all_signature_results.append(sig_result)
                
                if not sig_result["all_signed"]:
                    unsigned_pages.append({
                        "page": page_num,
                        "unsigned_codes": sig_result["unsigned_codes"],
                        "details": sig_result["details"]
                    })
        
        # Decision point
        if unsigned_pages:
            # Tenant hasn't signed - flag the document
            jobs[job_id] = {
                "status": "flagged",
                "reason": "tenant_signature_missing",
                "message": "Tenant signature not detected on one or more pages",
                "unsigned_pages": unsigned_pages,
                "all_results": all_signature_results
            }
            return
        
        # All tenant signatures present - proceed with landlord signing
        # ... your existing signing code ...
        
        jobs[job_id] = {
            "status": "complete",
            "download_url": f"/download/{job_id}",
            "signatures_applied": len(landlord_codes_found),
            "tenant_verification": all_signature_results
        }
        
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}
```

#### Add new endpoint for verification-only mode:

```python
@app.post("/verify-signatures")
async def verify_signatures(pdf: UploadFile = File(...)):
    """
    Verify if tenant has signed the document without applying landlord signatures.
    Returns immediately (no background processing needed for just verification).
    """
    pdf_bytes = await pdf.read()
    
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)
        results = []
        all_signed = True
        
        for page_num, page_image in enumerate(images, 1):
            decoded = decode(page_image, timeout=10000)
            detected_codes = [
                (code.data.decode('utf-8'),
                 (code.rect.left, code.rect.top, code.rect.width, code.rect.height))
                for code in decoded
            ]
            
            tenant_codes = [c for c, _ in detected_codes if c.startswith('TS_')]
            
            if tenant_codes:
                sig_result = process_page_for_signatures(
                    page_image,
                    detected_codes,
                    page_num
                )
                results.append(sig_result)
                
                if not sig_result["all_signed"]:
                    all_signed = False
        
        return {
            "all_signed": all_signed,
            "pages_checked": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## API Response Examples

### Successful signing (all tenant signatures detected):

```json
{
  "job_id": "abc123",
  "status": "complete",
  "download_url": "/download/abc123",
  "signatures_applied": 5,
  "tenant_verification": [
    {
      "page_number": 6,
      "is_phone_scan": false,
      "tenant_codes_found": ["TS_MAIN"],
      "all_signed": true,
      "unsigned_codes": [],
      "details": [
        {
          "code": "TS_MAIN",
          "status": "signed",
          "confidence": 0.85,
          "pixel_density": 0.072
        }
      ]
    }
  ]
}
```

### Flagged (tenant signature missing):

```json
{
  "job_id": "abc123",
  "status": "flagged",
  "reason": "tenant_signature_missing",
  "message": "Tenant signature not detected on one or more pages",
  "unsigned_pages": [
    {
      "page": 6,
      "unsigned_codes": ["TS_MAIN"],
      "details": [
        {
          "code": "TS_MAIN",
          "status": "unsigned",
          "confidence": 0.95,
          "pixel_density": 0.008
        }
      ]
    }
  ]
}
```

### Verification-only response:

```json
{
  "all_signed": false,
  "pages_checked": 5,
  "results": [
    {
      "page_number": 6,
      "is_phone_scan": true,
      "tenant_codes_found": ["TS_MAIN"],
      "all_signed": false,
      "unsigned_codes": ["TS_MAIN"],
      "details": [...]
    }
  ]
}
```

## n8n Workflow Updates

Update your n8n workflow to handle the new "flagged" status:

```
1. POST /sign → get job_id
2. Poll /status/{job_id}
3. Check status:
   - "complete" → download and continue
   - "flagged" → route to manual review / notify user
   - "error" → handle error
```

## Calibration

The detection thresholds in `signature_detection.py` are initial estimates. To calibrate:

1. Run the `/verify-signatures` endpoint on your sample files
2. Check the `pixel_density` values in the results
3. Adjust these constants in `DetectionConfig`:
   - `UNSIGNED_MAX_DENSITY` - increase if getting false positives on blank areas
   - `SIGNED_MIN_DENSITY` - decrease if missing faint signatures
   - `GRAYSCALE_THRESHOLD` - adjust for different ink colors/scan quality

### Expected pixel density ranges (initial estimates):

| Document Type | Unsigned | Signed |
|---------------|----------|--------|
| Digital PDF | 0.00 - 0.02 | 0.05 - 0.15 |
| Phone scan | 0.01 - 0.03 | 0.06 - 0.20 |

## Testing

Test with your sample files:
- `612_Unsigned.pdf` - should return `all_signed: false` for all TS_* codes
- `612_Signed_1.pdf` - should return `all_signed: true`
- `612_Signed_2.pdf` - should return `all_signed: true`
- `612_Signed_Phone_Scaned.pdf` - should return `all_signed: true` (with `is_phone_scan: true`)

## Troubleshooting

### False negatives (signed but detected as unsigned):
- Lower `SIGNED_MIN_DENSITY`
- Check if signature region offset is correct (signature might be in different position)
- Increase `REGION_WIDTH` to capture more of the signature area

### False positives (unsigned but detected as signed):
- Raise `UNSIGNED_MAX_DENSITY`
- Check for noise/artifacts in scan
- The `is_phone_scan` detection might need adjustment

### Region misalignment:
- Adjust `REGION_OFFSET_X` if signatures are positioned differently
- The detection assumes signatures are to the RIGHT of the DataMatrix
