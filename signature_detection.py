"""
Signature Detection Module for Lease Signer API

This module detects whether a tenant has signed (handwritten mark) next to
their DataMatrix signature marker codes (TS_*).

Logic:
1. Find all TS_* DataMatrix codes on the page
2. For each TS_* code, analyze a region to the RIGHT of the code
3. Check pixel density in that region - handwritten signatures have more
   dark pixels than a blank area
4. Return detection results per code

Based on sample analysis:
- Unsigned pages: Only DataMatrix code present (clean area to the right)
- Signed pages: DataMatrix code + handwritten mark to the right
"""

from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SignatureStatus(Enum):
    SIGNED = "signed"
    UNSIGNED = "unsigned"
    UNCERTAIN = "uncertain"


@dataclass
class SignatureDetectionResult:
    code: str
    status: SignatureStatus
    confidence: float  # 0.0 to 1.0
    pixel_density: float  # Percentage of dark pixels in detection region
    region_coords: Tuple[int, int, int, int]  # x1, y1, x2, y2 of analyzed region


@dataclass
class PageDetectionResult:
    page_number: int
    tenant_codes_found: List[str]
    detection_results: List[SignatureDetectionResult]
    all_signed: bool
    unsigned_codes: List[str]


# Detection parameters - calibrate these based on sample analysis
class DetectionConfig:
    # Region to analyze (relative to DataMatrix code position)
    # The handwritten signature appears to the RIGHT of the DataMatrix
    REGION_OFFSET_X = 10  # pixels to the right of code's right edge
    REGION_WIDTH = 80  # width of detection region in pixels
    REGION_HEIGHT_FACTOR = 1.5  # height relative to DataMatrix height
    
    # Thresholds for signature detection
    # These need calibration with actual samples
    GRAYSCALE_THRESHOLD = 180  # pixels darker than this are "ink"
    
    # Pixel density thresholds
    UNSIGNED_MAX_DENSITY = 0.02  # Below this = definitely unsigned (just noise)
    SIGNED_MIN_DENSITY = 0.05   # Above this = definitely signed
    # Between these values = uncertain
    
    # For phone scans (lower quality, more noise)
    PHONE_SCAN_NOISE_FACTOR = 1.5  # Multiply thresholds by this for phone scans


def detect_signature_region(
    page_image: Image.Image,
    code_rect: Tuple[int, int, int, int],  # x, y, width, height (y from bottom in PDF coords)
    image_height: int,
    config: DetectionConfig = None
) -> Tuple[float, Tuple[int, int, int, int]]:
    """
    Analyze the region to the right of a DataMatrix code for handwritten signature.
    
    Args:
        page_image: PIL Image of the page
        code_rect: Bounding box of the DataMatrix code (x, y, width, height)
                   Note: y is from bottom (PDF coordinates)
        image_height: Total height of the image (for coordinate conversion)
        config: Detection configuration parameters
    
    Returns:
        Tuple of (pixel_density, region_coords)
    """
    if config is None:
        config = DetectionConfig()
    
    code_x, code_y_from_bottom, code_width, code_height = code_rect
    
    # Convert from PDF coordinates (bottom-left origin) to image coordinates (top-left origin)
    code_y_from_top = image_height - code_y_from_bottom - code_height
    
    # Define the detection region (to the RIGHT of the DataMatrix)
    region_x1 = code_x + code_width + config.REGION_OFFSET_X
    region_y1 = int(code_y_from_top - (code_height * (config.REGION_HEIGHT_FACTOR - 1) / 2))
    region_x2 = region_x1 + config.REGION_WIDTH
    region_y2 = int(region_y1 + code_height * config.REGION_HEIGHT_FACTOR)
    
    # Clamp to image bounds
    region_x1 = max(0, region_x1)
    region_y1 = max(0, region_y1)
    region_x2 = min(page_image.width, region_x2)
    region_y2 = min(page_image.height, region_y2)
    
    # Extract the region
    region = page_image.crop((region_x1, region_y1, region_x2, region_y2))
    
    # Convert to grayscale
    gray = region.convert('L')
    
    # Convert to numpy array
    pixels = np.array(gray)
    
    # Count dark pixels (potential ink)
    dark_pixels = np.sum(pixels < config.GRAYSCALE_THRESHOLD)
    total_pixels = pixels.size
    
    pixel_density = dark_pixels / total_pixels if total_pixels > 0 else 0
    
    return pixel_density, (region_x1, region_y1, region_x2, region_y2)


def classify_signature(
    pixel_density: float,
    config: DetectionConfig = None,
    is_phone_scan: bool = False
) -> Tuple[SignatureStatus, float]:
    """
    Classify whether a signature is present based on pixel density.
    
    Returns:
        Tuple of (status, confidence)
    """
    if config is None:
        config = DetectionConfig()
    
    # Adjust thresholds for phone scans (more noise)
    unsigned_max = config.UNSIGNED_MAX_DENSITY
    signed_min = config.SIGNED_MIN_DENSITY
    
    if is_phone_scan:
        unsigned_max *= config.PHONE_SCAN_NOISE_FACTOR
        signed_min *= config.PHONE_SCAN_NOISE_FACTOR
    
    if pixel_density <= unsigned_max:
        # Clearly unsigned
        confidence = 1.0 - (pixel_density / unsigned_max)
        return SignatureStatus.UNSIGNED, min(1.0, confidence)
    
    elif pixel_density >= signed_min:
        # Clearly signed
        # Higher density = higher confidence (up to a point)
        confidence = min(1.0, (pixel_density - signed_min) / signed_min + 0.7)
        return SignatureStatus.SIGNED, confidence
    
    else:
        # Uncertain range
        # Calculate how close to each threshold
        range_size = signed_min - unsigned_max
        position = (pixel_density - unsigned_max) / range_size
        
        if position < 0.5:
            return SignatureStatus.UNCERTAIN, 0.5 - position
        else:
            return SignatureStatus.UNCERTAIN, position - 0.5


def detect_tenant_signatures(
    page_image: Image.Image,
    detected_codes: List[Tuple[str, Tuple[int, int, int, int]]],  # List of (code_value, rect)
    image_height: int,
    config: DetectionConfig = None,
    is_phone_scan: bool = False
) -> PageDetectionResult:
    """
    Detect tenant signatures on a page.
    
    Args:
        page_image: PIL Image of the rendered PDF page
        detected_codes: List of (code_value, bounding_rect) from DataMatrix detection
        image_height: Height of the image
        config: Detection configuration
        is_phone_scan: Whether this is a phone-scanned document (affects thresholds)
    
    Returns:
        PageDetectionResult with detection results for all TS_* codes
    """
    if config is None:
        config = DetectionConfig()
    
    # Filter to only tenant signature codes
    tenant_codes = [
        (code, rect) for code, rect in detected_codes 
        if code.startswith('TS_')
    ]
    
    results = []
    unsigned_codes = []
    
    for code_value, code_rect in tenant_codes:
        # Analyze the signature region
        pixel_density, region_coords = detect_signature_region(
            page_image, code_rect, image_height, config
        )
        
        # Classify
        status, confidence = classify_signature(pixel_density, config, is_phone_scan)
        
        result = SignatureDetectionResult(
            code=code_value,
            status=status,
            confidence=confidence,
            pixel_density=pixel_density,
            region_coords=region_coords
        )
        results.append(result)
        
        if status != SignatureStatus.SIGNED:
            unsigned_codes.append(code_value)
    
    return PageDetectionResult(
        page_number=0,  # Set by caller
        tenant_codes_found=[code for code, _ in tenant_codes],
        detection_results=results,
        all_signed=len(unsigned_codes) == 0 and len(tenant_codes) > 0,
        unsigned_codes=unsigned_codes
    )


def detect_if_phone_scan(page_image: Image.Image) -> bool:
    """
    Heuristic to detect if a page is from a phone scan.
    
    Phone scans typically have:
    - More noise/grain
    - Slight color cast
    - Lower contrast
    - Sometimes visible page edges/shadows
    
    This is a simple heuristic - can be improved with more samples.
    """
    # Convert to grayscale
    gray = np.array(page_image.convert('L'))
    
    # Check for characteristics of phone scans:
    
    # 1. Higher standard deviation in "white" areas (noise)
    # Sample the corners (usually white/blank)
    h, w = gray.shape
    corner_size = min(50, h // 10, w // 10)
    
    corners = [
        gray[:corner_size, :corner_size],  # top-left
        gray[:corner_size, -corner_size:],  # top-right
        gray[-corner_size:, :corner_size],  # bottom-left
        gray[-corner_size:, -corner_size:],  # bottom-right
    ]
    
    corner_stds = [np.std(corner) for corner in corners]
    avg_corner_std = np.mean(corner_stds)
    
    # Phone scans typically have std > 5 in "white" areas
    # Clean digital PDFs have std < 2
    if avg_corner_std > 4:
        return True
    
    # 2. Check if the "white" isn't actually white (color cast)
    corner_means = [np.mean(corner) for corner in corners]
    avg_corner_mean = np.mean(corner_means)
    
    # Clean PDFs have white backgrounds ~255
    # Phone scans often have gray-ish backgrounds ~230-250
    if avg_corner_mean < 245:
        return True
    
    return False


# Integration function for the main API
def process_page_for_signatures(
    page_image: Image.Image,
    detected_codes: List[Tuple[str, Tuple[int, int, int, int]]],
    page_number: int
) -> Dict:
    """
    Process a single page and return signature detection results.
    
    This is the main entry point for integration with the API.
    
    Args:
        page_image: PIL Image of the rendered PDF page
        detected_codes: List of (code_value, bounding_rect) from pylibdmtx
        page_number: 1-indexed page number
    
    Returns:
        Dictionary with detection results suitable for API response
    """
    image_height = page_image.height
    
    # Detect if this is a phone scan
    is_phone_scan = detect_if_phone_scan(page_image)
    
    # Run detection
    result = detect_tenant_signatures(
        page_image, 
        detected_codes, 
        image_height,
        is_phone_scan=is_phone_scan
    )
    result.page_number = page_number
    
    return {
        "page_number": page_number,
        "is_phone_scan": is_phone_scan,
        "tenant_codes_found": result.tenant_codes_found,
        "all_signed": result.all_signed,
        "unsigned_codes": result.unsigned_codes,
        "details": [
            {
                "code": r.code,
                "status": r.status.value,
                "confidence": round(r.confidence, 3),
                "pixel_density": round(r.pixel_density, 4),
                "region": r.region_coords
            }
            for r in result.detection_results
        ]
    }


# Example usage in main.py:
"""
# In your existing code where you detect DataMatrix codes:

from signature_detection import process_page_for_signatures

# After rendering page to image and detecting codes:
# page_image = ...  # PIL Image
# detected_codes = [(code.data.decode(), code.rect) for code in decoded]

signature_results = process_page_for_signatures(
    page_image,
    detected_codes,
    page_number
)

if not signature_results["all_signed"]:
    # Flag this document - tenant hasn't signed all required locations
    unsigned = signature_results["unsigned_codes"]
    # Handle accordingly...
else:
    # Proceed with landlord signature application
    pass
"""
