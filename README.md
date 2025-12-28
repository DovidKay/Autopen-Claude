# Lease Signer API

A microservice that detects DataMatrix codes in lease PDFs and places signature images at marked locations.

## DataMatrix Codes Supported

**Tenant Signatures:**
- `TS_MAIN` - Main lease signature
- `TS_RIDER_A` - Rider A signature
- `TS_RIDER_B` - Rider B signature
- `TS_RIDER_C` - Rider C signature
- `TS_BEDBUG` - Bedbug disclosure signature
- `TS_WIND` - Window guard notice signature

**Landlord Signatures:**
- `LS_MAIN` - Main lease signature
- `LS_RIDER_A` - Rider A signature
- `LS_RIDER_B` - Rider B signature
- `LS_RIDER_C` - Rider C signature
- `LS_BEDBUG` - Bedbug disclosure signature

## Deploy to Render

### Option 1: Using render.yaml (Recommended)

1. Push this folder to a GitHub/GitLab repo
2. In Render dashboard, click "New" → "Blueprint"
3. Connect your repo
4. Render will auto-detect `render.yaml` and deploy

### Option 2: Manual Docker Deployment

1. Push to GitHub/GitLab
2. In Render, click "New" → "Web Service"
3. Connect your repo
4. Select "Docker" as runtime
5. Deploy

## API Endpoints

### `GET /`
Health check. Returns `{"status": "healthy"}`.

### `POST /scan`
Scan a PDF and return all DataMatrix codes found (for testing/debugging).

**Request:**
- `pdf_file`: PDF file (multipart/form-data)
- `dpi`: Scan resolution (default: 200)

**Response:**
```json
{
  "total_pages": 13,
  "total_codes_found": 15,
  "signature_markers": [
    {"data": "TS_MAIN", "page": 6, "x": 85.2, "y": 142.5, ...},
    ...
  ],
  "metadata_codes": [...]
}
```

### `POST /sign`
Sign a PDF and return the signed PDF file.

**Request (multipart/form-data):**
- `pdf_file`: The lease PDF
- `tenant_signature`: PNG image for tenant signature (optional)
- `landlord_signature`: PNG image for landlord signature (optional)
- `dpi`: Scan resolution (default: 200)
- `sig_width`: Signature width in points (default: 100)
- `sig_height`: Signature height in points (default: 40)
- `x_offset`: Horizontal offset from DataMatrix (default: 0)
- `y_offset`: Vertical offset from DataMatrix (default: -5)

**Response:** Binary PDF file

### `POST /sign-with-details`
Same as `/sign` but returns JSON with base64-encoded PDF and metadata.

**Response:**
```json
{
  "success": true,
  "total_pages": 13,
  "signatures_applied": [
    {"page": 6, "code": "TS_MAIN", "x": 85.2, "y": 142.5},
    ...
  ],
  "signatures_count": 9,
  "pdf_base64": "JVBERi0xLjQK..."
}
```

## Usage from n8n

### Example: Sign a lease with HTTP Request node

1. **HTTP Request Node** configured as:
   - Method: `POST`
   - URL: `https://your-render-url.onrender.com/sign`
   - Body Content Type: `Form-Data/Multipart`
   - Body Parameters:
     - `pdf_file` (File) → from previous node or binary input
     - `tenant_signature` (File) → your tenant signature PNG
     - `landlord_signature` (File) → your landlord signature PNG
     - `sig_width` (Number) → 100
     - `sig_height` (Number) → 40

2. The response will be the signed PDF binary

### Example: Using /sign-with-details for more control

Use this endpoint when you need to know exactly what was signed:

```javascript
// In n8n Code node, after HTTP Request
const response = $json;
if (response.signatures_count !== 9) {
  throw new Error(`Expected 9 signatures, got ${response.signatures_count}`);
}
// Decode PDF from base64
const pdfBuffer = Buffer.from(response.pdf_base64, 'base64');
```

## Tuning Signature Placement

If signatures aren't landing in the right spot:

1. **Use `/scan` first** to see exact coordinates of DataMatrix codes
2. Adjust `x_offset` and `y_offset` to move signature relative to the code
3. Adjust `sig_width` and `sig_height` to resize the signature

Positive `x_offset` moves right, positive `y_offset` moves up (PDF coordinates).

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# On macOS, you also need:
brew install libdmtx poppler

# On Ubuntu/Debian:
apt-get install libdmtx0b libdmtx-dev poppler-utils

# Run locally
python main.py
# API available at http://localhost:8000
```

## Testing with curl

```bash
# Health check
curl http://localhost:8000/

# Scan a PDF
curl -X POST http://localhost:8000/scan \
  -F "pdf_file=@lease.pdf"

# Sign a PDF
curl -X POST http://localhost:8000/sign \
  -F "pdf_file=@lease.pdf" \
  -F "tenant_signature=@tenant_sig.png" \
  -F "landlord_signature=@landlord_sig.png" \
  -o signed_lease.pdf
```
