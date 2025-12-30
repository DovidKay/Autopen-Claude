"""
Signature Detection Module for Lease Signer API - V8

Problem: The signature LINE density varies between PDF versions.
The unsigned PDF has a thicker/darker line than the scanned versions.

New approach: Detect VERTICAL VARIATION in ink patterns.
- A straight signature line has ink only in a narrow horizontal band
- A handwritten signature has ink distributed across multiple Y positions

We analyze the VERTICAL DISTRIBUTION of dark pixels:
- Calculate the standard deviation of Y-positions of dark pixels
- High std dev = ink is spread vertically = signature present
- Low std dev = ink is in a narrow band = just the line
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
    confidence: float
    pixel_density: float
    region_coords: Tuple[int, int, int, int]


@dataclass
class PageDetectionResult:
    page_number: int
    tenant_codes_found: List[str]
    detection_results: List[SignatureDetectionResult]
    all_signed: bool
    unsigned_codes: List[str]


class DetectionConfig:
    # Region settings
    REGION_OFFSET_X = 50    # Start closer to catch full signature
    REGION_WIDTH = 180      # Wide region
    REGION_OFFSET_Y = -10   # Start slightly above DataMatrix
    REGION_HEIGHT = 65      # Taller to capture vertical spread
    
    GRAYSCALE_THRESHOLD = 180
    
    # Vertical spread thresholds
    # A straight line has low vertical std dev (maybe 2-5 pixels)
    # A signature curve has high vertical std dev (10+ pixels)
    UNSIGNED_MAX_VERTICAL_STD = 6.0   # Just a straight line
    SIGNED_MIN_VERTICAL_STD = 10.0    # Has curves
    
    PHONE_SCAN_NOISE_FACTOR = 1.3


def detect_signature_region(
    page_image: Image.Image,
    code_rect: Tuple[int, int, int, int],
    image_height: int,
    config: DetectionConfig = None
) -> Tuple[float, Tuple[int, int, int, int], float]:
    """
    Returns: (pixel_density, region_coords, vertical_std)
    """
    if config is None:
        config = DetectionConfig()
    
    code_x, code_y_from_top, code_width, code_height = code_rect
    
    region_x1 = code_x + code_width + config.REGION_OFFSET_X
    region_y1 = code_y_from_top + config.REGION_OFFSET_Y
    region_x2 = region_x1 + config.REGION_WIDTH
    region_y2 = region_y1 + config.REGION_HEIGHT
    
    region_x1 = max(0, region_x1)
    region_y1 = max(0, region_y1)
    region_x2 = min(page_image.width, region_x2)
    region_y2 = min(page_image.height, region_y2)
    
    region = page_image.crop((region_x1, region_y1, region_x2, region_y2))
    gray = region.convert('L')
    pixels = np.array(gray)
    
    # Find dark pixels
    dark_mask = pixels < config.GRAYSCALE_THRESHOLD
    dark_pixels_count = np.sum(dark_mask)
    total_pixels = pixels.size
    pixel_density = dark_pixels_count / total_pixels if total_pixels > 0 else 0
    
    # Calculate vertical distribution of dark pixels
    # Get Y coordinates of all dark pixels
    dark_y_coords = np.where(dark_mask)[0]  # Row indices = Y coordinates
    
    if len(dark_y_coords) > 10:  # Need enough pixels for meaningful std
        vertical_std = np.std(dark_y_coords)
    else:
        vertical_std = 0.0  # Not enough ink to analyze
    
    return pixel_density, (region_x1, region_y1, region_x2, region_y2), vertical_std


def classify_signature(
    pixel_density: float,
    vertical_std: float,
    config: DetectionConfig = None,
    is_phone_scan: bool = False
) -> Tuple[SignatureStatus, float]:
    if config is None:
        config = DetectionConfig()
    
    unsigned_max = config.UNSIGNED_MAX_VERTICAL_STD
    signed_min = config.SIGNED_MIN_VERTICAL_STD
    
    if is_phone_scan:
        unsigned_max *= config.PHONE_SCAN_NOISE_FACTOR
        signed_min *= config.PHONE_SCAN_NOISE_FACTOR
    
    # Very little ink = unsigned (regardless of spread)
    if pixel_density < 0.005:
        return SignatureStatus.UNSIGNED, 0.9
    
    # Use vertical spread as primary classifier
    if vertical_std <= unsigned_max:
        confidence = 1.0 - (vertical_std / unsigned_max) if unsigned_max > 0 else 1.0
        return SignatureStatus.UNSIGNED, min(1.0, confidence)
    elif vertical_std >= signed_min:
        confidence = min(1.0, (vertical_std - signed_min) / signed_min + 0.7)
        return SignatureStatus.SIGNED, confidence
    else:
        range_size = signed_min - unsigned_max
        position = (vertical_std - unsigned_max) / range_size if range_size > 0 else 0.5
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
    if config is None:
        config = DetectionConfig()
    
    tenant_codes = [
        (code, rect) for code, rect in detected_codes 
        if code.startswith('TS_')
    ]
    
    results = []
    unsigned_codes = []
    
    for code_value, code_rect in tenant_codes:
        pixel_density, region_coords, vertical_std = detect_signature_region(
            page_image, code_rect, image_height, config
        )
        
        status, confidence = classify_signature(pixel_density, vertical_std, config, is_phone_scan)
        
        result = SignatureDetectionResult(
            code=code_value,
            status=status,
            confidence=confidence,
            pixel_density=vertical_std,  # Store vertical_std in pixel_density field for debugging
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
                "pixel_density": round(r.pixel_density, 4),  # This is actually vertical_std now
                "region": r.region_coords
            }
            for r in result.detection_results
        ]
    }
