import os
import cv2
import numpy as np
import random
import csv
import shutil

# Ensure directories exist
os.makedirs("dataset/genuine", exist_ok=True)
os.makedirs("dataset/forged", exist_ok=True)

def generate_random_signature(width, height, color=(20, 30, 120), thickness=2):
    """
    Generates a synthetic handwritten signature using overlapping bezier-like curves.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw several overlapping strokes to make it look like a signature
    num_strokes = random.randint(3, 6)
    for _ in range(num_strokes):
        # Generate control points for a bezier curve
        pts = []
        num_pts = random.randint(3, 5)
        for i in range(num_pts):
            if i == 0:
                x = random.randint(10, int(width * 0.3))
                y = random.randint(10, height - 10)
            elif i == num_pts - 1:
                x = random.randint(int(width * 0.7), width - 10)
                y = random.randint(10, height - 10)
            else:
                x = random.randint(int(width * 0.2), int(width * 0.8))
                y = random.randint(10, height - 10)
            pts.append((x, y))
            
        # Draw curve by interpolating between control points
        curve_pts = []
        for t in np.linspace(0, 1, 100):
            # De Casteljau's algorithm for bezier curve
            temp_pts = list(pts)
            while len(temp_pts) > 1:
                next_pts = []
                for j in range(len(temp_pts) - 1):
                    x_c = int((1 - t) * temp_pts[j][0] + t * temp_pts[j+1][0])
                    y_c = int((1 - t) * temp_pts[j][1] + t * temp_pts[j+1][1])
                    next_pts.append((x_c, y_c))
                temp_pts = next_pts
            curve_pts.append(temp_pts[0])
            
        curve_pts = np.array(curve_pts, dtype=np.int32)
        cv2.polylines(img, [curve_pts], False, color, thickness, cv2.LINE_AA)
        
    # Add a horizontal underline flourish
    if random.choice([True, False]):
        start_pt = (random.randint(5, int(width * 0.2)), random.randint(int(height * 0.7), height - 5))
        end_pt = (random.randint(int(width * 0.8), width - 5), random.randint(int(height * 0.7), height - 5))
        cv2.line(img, start_pt, end_pt, color, thickness, cv2.LINE_AA)
        
    return img

def create_document_page(width=800, height=1000):
    """
    Creates a synthetic document background page with text lines.
    """
    # Soft off-white paper color
    paper_color = (random.randint(250, 255), random.randint(250, 255), random.randint(245, 255))
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:, :] = paper_color
    
    # Draw some standard document headings and text lines
    cv2.putText(img, "AGREEMENT OF TERMS AND CONDITIONS", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.putText(img, "Date: June 3, 2026", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Draw paragraphs as gray rectangles or generic text
    y_start = 180
    for i in range(12):
        # Draw simulated text lines
        line_w = random.randint(500, 700)
        cv2.line(img, (50, y_start), (50 + line_w, y_start), (80, 80, 80), 2)
        y_start += 35
        if i % 4 == 3:
            y_start += 20  # paragraph spacing
            
    cv2.putText(img, "IN WITNESS WHEREOF, the parties hereto have executed this Agreement.", (50, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(img, "Authorized Signature:", (50, y_start + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    cv2.line(img, (50, y_start + 180), (350, y_start + 180), (120, 120, 120), 1)
    
    # Bounding box where the signature should be placed
    sig_box = {
        "x_min": 60,
        "y_min": y_start + 80,
        "x_max": 340,
        "y_max": y_start + 175
    }
    
    return img, sig_box

def apply_scan_artifacts(img, jpeg_quality=85, noise_level=3.0):
    """
    Applies scanning artifacts: Gaussian blur, scanning noise, and JPEG compression.
    """
    # 1. Subtle blur
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 2. Add Gaussian noise
    noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
    img_noisy = np.clip(img_blur.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 3. JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', img_noisy, encode_param)
    img_compressed = cv2.imdecode(encimg, 1)
    
    return img_compressed

def rotate_image(image, angle):
    """
    Rotates an image by a given angle (in degrees) with white background padding.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result

def generate_dataset(num_samples=100):
    print(f"Generating synthetic dataset: {num_samples} genuine and {num_samples} forged...")
    metadata = []
    
    for i in range(num_samples):
        # ----------------------------------------------------
        # Generate Genuine Sample
        # ----------------------------------------------------
        doc_gen, sig_box = create_document_page()
        
        # Draw signature directly on the document (genuine scanned signature)
        sig_w = sig_box["x_max"] - sig_box["x_min"]
        sig_h = sig_box["y_max"] - sig_box["y_min"]
        sig_img = generate_random_signature(sig_w, sig_h)
        
        # Place signature on document (multiply to blend ink with background paper)
        sig_region = doc_gen[sig_box["y_min"]:sig_box["y_max"], sig_box["x_min"]:sig_box["x_max"]]
        blended = cv2.multiply(sig_region.astype(float)/255.0, sig_img.astype(float)/255.0)
        doc_gen[sig_box["y_min"]:sig_box["y_max"], sig_box["x_min"]:sig_box["x_max"]] = (blended * 255).astype(np.uint8)
        
        # Apply standard single scan artifacts
        doc_gen_scanned = apply_scan_artifacts(doc_gen, jpeg_quality=85, noise_level=3.0)
        
        gen_filename = f"gen_{i+1:03d}.jpg"
        gen_filepath = os.path.join("dataset/genuine", gen_filename)
        cv2.imwrite(gen_filepath, doc_gen_scanned)
        
        metadata.append([
            f"genuine/{gen_filename}", 
            0, # Label: Genuine
            sig_box["x_min"], 
            sig_box["y_min"], 
            sig_box["x_max"], 
            sig_box["y_max"]
        ])
        
        # ----------------------------------------------------
        # Generate Forged Sample (Digitally Inserted Signature)
        # ----------------------------------------------------
        doc_forg, sig_box_forg = create_document_page()
        
        # Generate signature on a SEPARATE document
        sig_w = sig_box_forg["x_max"] - sig_box_forg["x_min"]
        sig_h = sig_box_forg["y_max"] - sig_box_forg["y_min"]
        sig_img_src = generate_random_signature(sig_w, sig_h)
        
        # Simulate scanning of the SOURCE document (with lower JPEG quality, e.g. 70, representing different history)
        sig_img_scanned = apply_scan_artifacts(sig_img_src, jpeg_quality=70, noise_level=4.0)
        
        # Resample the signature (random rotation and scaling to introduce interpolation artifacts)
        angle = random.uniform(-4, 4)
        scale = random.uniform(0.9, 1.1)
        
        # Rotate
        sig_resampled = rotate_image(sig_img_scanned, angle)
        # Scale
        new_w = int(sig_resampled.shape[1] * scale)
        new_h = int(sig_resampled.shape[0] * scale)
        sig_resampled = cv2.resize(sig_resampled, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to match the target box size
        if sig_resampled.shape[0] > sig_h or sig_resampled.shape[1] > sig_w:
            # Crop
            y_off = (sig_resampled.shape[0] - sig_h) // 2
            x_off = (sig_resampled.shape[1] - sig_w) // 2
            sig_final = sig_resampled[y_off:y_off+sig_h, x_off:x_off+sig_w]
        else:
            # Pad
            sig_final = np.ones((sig_h, sig_w, 3), dtype=np.uint8) * 255
            y_off = (sig_h - sig_resampled.shape[0]) // 2
            x_off = (sig_w - sig_resampled.shape[1]) // 2
            sig_final[y_off:y_off+sig_resampled.shape[0], x_off:x_off+sig_resampled.shape[1]] = sig_resampled
            
        # Paste the signature onto the target document
        sig_region_forg = doc_forg[sig_box_forg["y_min"]:sig_box_forg["y_max"], sig_box_forg["x_min"]:sig_box_forg["x_max"]]
        
        # Simple blend: if it's a forgery, we can paste it.
        # To simulate realistic paste boundaries, we can either do a sharp paste or slight border blending.
        # We will do a direct blend (multiply) but because the signature came from a different scan background,
        # it has background mismatch.
        blended_forg = cv2.multiply(sig_region_forg.astype(float)/255.0, sig_final.astype(float)/255.0)
        
        # Write back to document
        doc_forg[sig_box_forg["y_min"]:sig_box_forg["y_max"], sig_box_forg["x_min"]:sig_box_forg["x_max"]] = (blended_forg * 255).astype(np.uint8)
        
        # Now apply the TARGET document's scan artifacts (e.g. quality 95, representing the second scanning)
        doc_forg_scanned = apply_scan_artifacts(doc_forg, jpeg_quality=95, noise_level=2.0)
        
        forg_filename = f"forg_{i+1:03d}.jpg"
        forg_filepath = os.path.join("dataset/forged", forg_filename)
        cv2.imwrite(forg_filepath, doc_forg_scanned)
        
        metadata.append([
            f"forged/{forg_filename}", 
            1, # Label: Forged
            sig_box_forg["x_min"], 
            sig_box_forg["y_min"], 
            sig_box_forg["x_max"], 
            sig_box_forg["y_max"]
        ])
        
    # Save metadata CSV
    with open("dataset_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "x_min", "y_min", "x_max", "y_max"])
        writer.writerows(metadata)
        
    print("Dataset generation completed.")

if __name__ == "__main__":
    # Generate 100 genuine and 100 forged samples (200 total) as planned
    generate_dataset(100)
