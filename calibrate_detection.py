#!/usr/bin/env python3
"""
Calibration script for signature detection.

Run this against your sample PDFs to:
1. See actual pixel density values for signed vs unsigned documents
2. Determine the right thresholds for your specific use case
3. Verify detection is working correctly

Usage:
    python calibrate_detection.py /path/to/sample.pdf [--page N]
    python calibrate_detection.py /path/to/folder/  # Process all PDFs in folder
"""

import sys
import os
import json
from pathlib import Path
from pdf2image import convert_from_path
from pylibdmtx.pylibdmtx import decode
from PIL import Image, ImageDraw
from signature_detection import (
    process_page_for_signatures,
    detect_signature_region,
    detect_if_phone_scan,
    DetectionConfig
)


def analyze_pdf(pdf_path: str, specific_page: int = None, save_debug_images: bool = True):
    """
    Analyze a PDF and return signature detection results.
    
    Args:
        pdf_path: Path to the PDF file
        specific_page: If set, only analyze this page (1-indexed)
        save_debug_images: If True, save images showing detection regions
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {pdf_path}")
    print(f"{'='*60}")
    
    # Convert PDF to images
    print("Converting PDF to images (150 DPI)...")
    try:
        images = convert_from_path(pdf_path, dpi=150)
    except Exception as e:
        print(f"ERROR: Could not convert PDF: {e}")
        return None
    
    print(f"Total pages: {len(images)}")
    
    results = {
        "file": os.path.basename(pdf_path),
        "total_pages": len(images),
        "pages_with_tenant_codes": [],
        "summary": {
            "signed_codes": [],
            "unsigned_codes": [],
            "uncertain_codes": []
        }
    }
    
    # Create debug output directory
    if save_debug_images:
        debug_dir = Path(pdf_path).stem + "_debug"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug images will be saved to: {debug_dir}/")
    
    # Process each page
    pages_to_process = [specific_page] if specific_page else range(1, len(images) + 1)
    
    for page_num in pages_to_process:
        if page_num > len(images):
            print(f"Page {page_num} does not exist (only {len(images)} pages)")
            continue
            
        page_image = images[page_num - 1]
        print(f"\n--- Page {page_num} ---")
        
        # Check if phone scan
        is_phone_scan = detect_if_phone_scan(page_image)
        print(f"Phone scan detected: {is_phone_scan}")
        
        # Detect DataMatrix codes
        print("Detecting DataMatrix codes...")
        try:
            decoded = decode(page_image, timeout=10000)
        except Exception as e:
            print(f"ERROR decoding page {page_num}: {e}")
            continue
        
        if not decoded:
            print("No DataMatrix codes found on this page")
            continue
        
        # Parse detected codes
        detected_codes = []
        for code in decoded:
            try:
                code_value = code.data.decode('utf-8')
                rect = (code.rect.left, code.rect.top, code.rect.width, code.rect.height)
                detected_codes.append((code_value, rect))
                print(f"  Found: {code_value[:30]}{'...' if len(code_value) > 30 else ''}")
                print(f"         Position: x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]}")
            except Exception as e:
                print(f"  Could not decode: {e}")
        
        # Filter for tenant codes
        tenant_codes = [(c, r) for c, r in detected_codes if c.startswith('TS_')]
        
        if not tenant_codes:
            print("No tenant signature codes (TS_*) on this page")
            continue
        
        print(f"\nTenant signature codes found: {[c for c, _ in tenant_codes]}")
        
        # Run signature detection
        sig_result = process_page_for_signatures(
            page_image,
            detected_codes,
            page_num
        )
        
        results["pages_with_tenant_codes"].append(sig_result)
        
        # Print detailed results
        print("\nSignature Detection Results:")
        print("-" * 40)
        for detail in sig_result["details"]:
            status_emoji = {
                "signed": "✓",
                "unsigned": "✗",
                "uncertain": "?"
            }.get(detail["status"], "?")
            
            print(f"  {status_emoji} {detail['code']}")
            print(f"    Status: {detail['status']}")
            print(f"    Confidence: {detail['confidence']:.1%}")
            print(f"    Pixel Density: {detail['pixel_density']:.4f} ({detail['pixel_density']*100:.2f}%)")
            print(f"    Detection Region: {detail['region']}")
            
            # Track in summary
            if detail["status"] == "signed":
                results["summary"]["signed_codes"].append({
                    "code": detail["code"],
                    "page": page_num,
                    "density": detail["pixel_density"]
                })
            elif detail["status"] == "unsigned":
                results["summary"]["unsigned_codes"].append({
                    "code": detail["code"],
                    "page": page_num,
                    "density": detail["pixel_density"]
                })
            else:
                results["summary"]["uncertain_codes"].append({
                    "code": detail["code"],
                    "page": page_num,
                    "density": detail["pixel_density"]
                })
        
        # Save debug image
        if save_debug_images and tenant_codes:
            save_debug_image(
                page_image,
                tenant_codes,
                sig_result["details"],
                f"{debug_dir}/page_{page_num}.png"
            )
    
    return results


def save_debug_image(page_image: Image.Image, tenant_codes, detection_details, output_path: str):
    """Save an annotated image showing detection regions."""
    
    # Make a copy to draw on
    debug_img = page_image.copy()
    draw = ImageDraw.Draw(debug_img)
    
    config = DetectionConfig()
    image_height = page_image.height
    
    for code_value, code_rect in tenant_codes:
        code_x, code_y_from_bottom, code_width, code_height = code_rect
        
        # Convert to image coordinates
        code_y_from_top = image_height - code_y_from_bottom - code_height
        
        # Draw DataMatrix bounding box (blue)
        draw.rectangle(
            [code_x, code_y_from_top, code_x + code_width, code_y_from_top + code_height],
            outline="blue",
            width=2
        )
        
        # Find the detection result for this code
        detail = next((d for d in detection_details if d["code"] == code_value), None)
        
        if detail:
            # Draw detection region
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
            
            # Add label
            label = f"{code_value}: {detail['status']} ({detail['pixel_density']:.4f})"
            draw.text((region[0], region[1] - 20), label, fill=color)
    
    # Save
    debug_img.save(output_path)
    print(f"  Debug image saved: {output_path}")


def print_summary(all_results: list):
    """Print a summary across all analyzed files."""
    
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    
    all_signed_densities = []
    all_unsigned_densities = []
    
    for result in all_results:
        if result is None:
            continue
            
        print(f"\n{result['file']}:")
        
        for item in result["summary"]["signed_codes"]:
            print(f"  SIGNED   - Page {item['page']}, {item['code']}: density = {item['density']:.4f}")
            all_signed_densities.append(item['density'])
            
        for item in result["summary"]["unsigned_codes"]:
            print(f"  UNSIGNED - Page {item['page']}, {item['code']}: density = {item['density']:.4f}")
            all_unsigned_densities.append(item['density'])
            
        for item in result["summary"]["uncertain_codes"]:
            print(f"  UNCERTAIN - Page {item['page']}, {item['code']}: density = {item['density']:.4f}")
    
    # Calculate recommended thresholds
    print("\n" + "-" * 60)
    print("DENSITY STATISTICS:")
    print("-" * 60)
    
    if all_unsigned_densities:
        print(f"Unsigned densities: min={min(all_unsigned_densities):.4f}, max={max(all_unsigned_densities):.4f}, avg={sum(all_unsigned_densities)/len(all_unsigned_densities):.4f}")
    else:
        print("No unsigned samples detected")
    
    if all_signed_densities:
        print(f"Signed densities:   min={min(all_signed_densities):.4f}, max={max(all_signed_densities):.4f}, avg={sum(all_signed_densities)/len(all_signed_densities):.4f}")
    else:
        print("No signed samples detected")
    
    # Recommend thresholds
    print("\n" + "-" * 60)
    print("RECOMMENDED THRESHOLDS:")
    print("-" * 60)
    
    if all_unsigned_densities and all_signed_densities:
        max_unsigned = max(all_unsigned_densities)
        min_signed = min(all_signed_densities)
        
        if max_unsigned < min_signed:
            # Clear separation - good!
            gap = min_signed - max_unsigned
            recommended_unsigned_max = max_unsigned + (gap * 0.3)
            recommended_signed_min = min_signed - (gap * 0.3)
            
            print(f"Good separation between signed and unsigned!")
            print(f"Gap between max unsigned ({max_unsigned:.4f}) and min signed ({min_signed:.4f}): {gap:.4f}")
            print(f"\nRecommended settings for DetectionConfig:")
            print(f"  UNSIGNED_MAX_DENSITY = {recommended_unsigned_max:.4f}")
            print(f"  SIGNED_MIN_DENSITY = {recommended_signed_min:.4f}")
        else:
            print(f"WARNING: Overlap between signed and unsigned densities!")
            print(f"Max unsigned: {max_unsigned:.4f}")
            print(f"Min signed: {min_signed:.4f}")
            print(f"\nYou may need to:")
            print(f"  1. Adjust REGION_OFFSET_X or REGION_WIDTH")
            print(f"  2. Use different thresholds for phone scans")
            print(f"  3. Accept some uncertainty in detection")
    else:
        print("Need both signed and unsigned samples for threshold recommendations")
    
    print("\n" + "=" * 60)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python calibrate_detection.py ./612_Unsigned.pdf")
        print("  python calibrate_detection.py ./samples/")
        sys.exit(1)
    
    path = sys.argv[1]
    specific_page = None
    
    # Check for --page argument
    if "--page" in sys.argv:
        page_idx = sys.argv.index("--page")
        if page_idx + 1 < len(sys.argv):
            specific_page = int(sys.argv[page_idx + 1])
    
    all_results = []
    
    if os.path.isdir(path):
        # Process all PDFs in directory
        pdf_files = sorted(Path(path).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in {path}")
        
        for pdf_file in pdf_files:
            result = analyze_pdf(str(pdf_file), specific_page)
            all_results.append(result)
    else:
        # Single file
        result = analyze_pdf(path, specific_page)
        all_results.append(result)
    
    # Print summary
    print_summary(all_results)
    
    # Save results to JSON
    output_file = "calibration_results.json"
    with open(output_file, "w") as f:
        json.dump([r for r in all_results if r], f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
