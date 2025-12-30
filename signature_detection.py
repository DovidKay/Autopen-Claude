"""
Signature Detection Module for Lease Signer API - V2

This module detects whether a tenant has signed (handwritten mark) next to
their DataMatrix signature marker codes (TS_*).

Key insight from calibration:
- The unsigned document has a printed signature LINE that creates dark pixels
- The signed document has handwritten marks ABOVE/ON the signature line
- We need to detect ink in the area ABOVE the signature line only

Logic:
1. Find all TS_* DataMatrix codes on the page
2. For each TS_* code, analyze a region to the RIGHT and ABOVE the code
   (where handwritten signatures appear, but not where the printed line is)
3. Check pixel density in that region
4. Return detection results per code
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
    # and ABOVE the printed signature line
    REGION_OFFSET_X = 10  # pixels to the right of code's right edge
    REGION_WIDTH = 200    # width of detection region in pixels
    
    # CRITICAL: We look ABOVE the DataMatrix code level
    # The signature line is at the same Y level as the code
    # Handwritten marks curve ABOVE this line
    REGION_OFFSET_Y = -50  # Start above the code (negative = up in image coords)
    REGION_HEIGHT = 45     # Height of detection region (just the area above the line)
    
    # Thresholds for signature detection
    GRAYSCALE_THRESHOLD = 180  # pixels darker than this are "ink"
    
    # Pixel density thresholds
    # In the "above line" region:
    # - Unsigned: very few dark pixels (just noise) ~0.01-0.02
    # - Signed: handwritten curves present ~0.03+
    UNSIGNED_MAX_DENSITY = 0.02   # Below this = definitely unsigned
    SIGNED_MIN_DENSITY = 0.03     # Above this = definitely signed
    
    # For phone scans (lower quality, more noise)
    PHONE_SCAN_NOISE_FACTOR = 1.5


def detect_signature_region(
    page_image: Image.Image,
    code_rect: Tuple[int, int, int, int],  # x, y, width, height from pylibdmtx
    image_height: int,
    config: DetectionConfig = None
) -> Tuple[float, Tuple[int, int, int, int]]:
    """
    Analyze the region ABOVE and to the right of a DataMatrix code for handwritten signature.
    
    The key insight is that:
    - The signature LINE is at the same Y level as the DataMatrix
    - Handwritten signatures curve ABOVE this line
    - So we look at the area above the code, to the right
    """
    if config is None:
        config = DetectionConfig()
    
    code_x, code_y_from_top, code_width, code_height = code_rect
    
    # Define the detection region:
    # - To the RIGHT of the DataMatrix
    # - ABOVE the DataMatrix level (where handwritten marks appear)
    region_x1 = code_x + code_width + config.REGION_OFFSET_X
    region_y1 = code_y_from_top + config.REGION_OFFSET_Y  # Above the code
    region_x2 = region_x1 + config.REGION_WIDTH
    region_y2 = region_y1 + config.REGION_HEIGHT
    
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
        confidence = 1.0 - (pixel_density / unsigned_max) if unsigned_max > 0 else 1.0
        return SignatureStatus.UNSIGNED, min(1.0, confidence)
    
    elif pixel_density >= signed_min:
        # Clearly signed
        confidence = min(1.0, (pixel_density - signed_min) / signed_min + 0.7)
        return SignatureStatus.SIGNED, confidence
    
    else:
        # Uncertain range
        range_size = signed_min - unsigned_max
        position = (pixel_density - unsigned_max) / range_size if range_size > 0 else 0.5
        
        if position < 0.5:
            return SignatureStatus.UNCERTAIN, 0.5 - position
        else:
            return SignatureStatus.UNCERTAIN, position - 0.5


def detect_tenant_signatures(
    page_image: Image.Image,
    detected_codes: List[Tuple[str, Tuple[int, int, int, int]]],
    image_height: int,
    config: DetectionConfig = None,
    is_phone_scan: bool = False
) -> PageDetectionResult:
    """
    Detect tenant signatures on a page.
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
        page_number=0,
        tenant_codes_found=[code for code, _ in tenant_codes],
        detection_results=results,
        all_signed=len(unsigned_codes) == 0 and len(tenant_codes) > 0,
        unsigned_codes=unsigned_codes
    )


def detect_if_phone_scan(page_image: Image.Image) -> bool:
    """
    Heuristic to detect if a page is from a phone scan.
    """
    gray = np.array(page_image.convert('L'))
    
    h, w = gray.shape
    corner_size = min(50, h // 10, w // 10)
    
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:],
    ]
    
    corner_stds = [np.std(corner) for corner in corners]
    avg_corner_std = np.mean(corner_stds)
    
    if avg_corner_std > 4:
        return True
    
    corner_means = [np.mean(corner) for corner in corners]
    avg_corner_mean = np.mean(corner_means)
    
    if avg_corner_mean < 245:
        return True
    
    return False


def process_page_for_signatures(
    page_image: Image.Image,
    detected_codes: List[Tuple[str, Tuple[int, int, int, int]]],
    page_number: int
) -> Dict:
    """
    Process a single page and return signature detection results.
    """
    image_height = page_image.height
    
    is_phone_scan = detect_if_phone_scan(page_image)
    
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
