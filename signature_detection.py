"""
Signature Detection Module for Lease Signer API - V3

Key insight from calibration:
- Looking above the signature line captures printed text
- Looking at the signature line captures the printed line itself
- We need to look at a specific spot where ONLY handwritten ink would appear

Strategy V3:
- Look at the area to the FAR RIGHT of the DataMatrix
- At the same Y level as the DataMatrix (not above, not below)
- This area should have:
  - Unsigned: just the thin signature line OR blank space
  - Signed: the curve of the handwritten signature
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
    # Look FAR to the right of the DataMatrix
    # Skip the first ~100px to avoid the start of the signature line
    REGION_OFFSET_X = 120  # Start further right
    REGION_WIDTH = 150     # Width of detection region
    
    # Look at a narrow band at the SAME level as the DataMatrix
    # (not above where printed text might be)
    REGION_OFFSET_Y = 5    # Slightly below top of DataMatrix
    REGION_HEIGHT = 35     # Narrow band
    
    GRAYSCALE_THRESHOLD = 180
    
    # Thresholds - will need calibration
    UNSIGNED_MAX_DENSITY = 0.02
    SIGNED_MIN_DENSITY = 0.04
    
    PHONE_SCAN_NOISE_FACTOR = 1.5


def detect_signature_region(
    page_image: Image.Image,
    code_rect: Tuple[int, int, int, int],
    image_height: int,
    config: DetectionConfig = None
) -> Tuple[float, Tuple[int, int, int, int]]:
    """
    Analyze a region far to the right of the DataMatrix for signature detection.
    """
    if config is None:
        config = DetectionConfig()
    
    code_x, code_y_from_top, code_width, code_height = code_rect
    
    # Define the detection region - far to the right, at same Y level
    region_x1 = code_x + code_width + config.REGION_OFFSET_X
    region_y1 = code_y_from_top + config.REGION_OFFSET_Y
    region_x2 = region_x1 + config.REGION_WIDTH
    region_y2 = region_y1 + config.REGION_HEIGHT
    
    # Clamp to image bounds
    region_x1 = max(0, region_x1)
    region_y1 = max(0, region_y1)
    region_x2 = min(page_image.width, region_x2)
    region_y2 = min(page_image.height, region_y2)
    
    # Extract and analyze the region
    region = page_image.crop((region_x1, region_y1, region_x2, region_y2))
    gray = region.convert('L')
    pixels = np.array(gray)
    
    dark_pixels = np.sum(pixels < config.GRAYSCALE_THRESHOLD)
    total_pixels = pixels.size
    
    pixel_density = dark_pixels / total_pixels if total_pixels > 0 else 0
    
    return pixel_density, (region_x1, region_y1, region_x2, region_y2)


def classify_signature(
    pixel_density: float,
    config: DetectionConfig = None,
    is_phone_scan: bool = False
) -> Tuple[SignatureStatus, float]:
    if config is None:
        config = DetectionConfig()
    
    unsigned_max = config.UNSIGNED_MAX_DENSITY
    signed_min = config.SIGNED_MIN_DENSITY
    
    if is_phone_scan:
        unsigned_max *= config.PHONE_SCAN_NOISE_FACTOR
        signed_min *= config.PHONE_SCAN_NOISE_FACTOR
    
    if pixel_density <= unsigned_max:
        confidence = 1.0 - (pixel_density / unsigned_max) if unsigned_max > 0 else 1.0
        return SignatureStatus.UNSIGNED, min(1.0, confidence)
    elif pixel_density >= signed_min:
        confidence = min(1.0, (pixel_density - signed_min) / signed_min + 0.7)
        return SignatureStatus.SIGNED, confidence
    else:
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
    if config is None:
        config = DetectionConfig()
    
    tenant_codes = [
        (code, rect) for code, rect in detected_codes 
        if code.startswith('TS_')
    ]
    
    results = []
    unsigned_codes = []
    
    for code_value, code_rect in tenant_codes:
        pixel_density, region_coords = detect_signature_region(
            page_image, code_rect, image_height, config
        )
        
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
                "pixel_density": round(r.pixel_density, 4),
                "region": r.region_coords
            }
            for r in result.detection_results
        ]
    }
