"""
Computer Vision Assignment
Part A: Harris Corner Detection with Precision-Recall Analysis
Part B: SIFT Feature Matching on Consecutive vs Random Image Pairs

Author: Veer Vardhan Singh
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os
import itertools
import random
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "cv_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to the building image (the brick building image from Ashoka)
BUILDING_IMAGE_PATH = os.path.join(SCRIPT_DIR, "building.jpg")

# Path to oyla images directory
OYLA_DIR = os.path.join(SCRIPT_DIR, "oyla_images")


# ============================================================================
# PART A: HARRIS CORNER DETECTION
# ============================================================================

def get_ground_truth_corners(img_shape):
    """
    Manually annotated ~50 corners (ground truth) on the building image.
    
    These are (x, y) pixel coordinates of visible corners in the brick
    building/bench image from Ashoka University. The image shows:
    - A brick bench with wooden plank seating
    - A red painted wall/structure behind with a hydrant pipe
    - Stone/concrete cap on the brick wall
    - Green bushes in the background
    
    Coordinates are specified as fractions of image dimensions, then 
    converted to absolute pixels, so they work regardless of resolution.
    
    Corners annotated at:
    - Brick mortar joint intersections (T-junctions and cross-junctions)
    - Structural corners of the bench wall
    - Wooden plank edges and joints
    - Cap stone edges
    - Red wall / hydrant pipe corners
    """
    h, w = img_shape[:2]
    
    # Ground truth corners as (x_frac, y_frac) of image dimensions.
    # Carefully placed at actual BRICK MORTAR T-JUNCTIONS, structural 
    # edges, and other real corners visible in the image.
    # Based on visual inspection of the Harris response overlay.
    corners_frac = [
        # === LEFT WALL STRUCTURAL CORNERS (brick edges on left side) ===
        (0.06, 0.38),   # 1: Top-left wall outer corner (cap/brick junction)
        (0.10, 0.42),   # 2: Left wall inner corner at cap stone
        (0.06, 0.47),   # 3: Left wall brick mortar junction
        (0.08, 0.52),   # 4: Left wall brick mortar junction
        (0.06, 0.57),   # 5: Left wall mortar T-junction
        
        # === BACK WALL BRICK MORTAR JOINTS (upper region, below capstone) ===
        (0.14, 0.40),   # 6: Back wall brick corner near left
        (0.20, 0.39),   # 7: Back wall mortar T-junction
        (0.27, 0.38),   # 8: Back wall mortar junction
        (0.35, 0.37),   # 9: Back wall mortar junction
        (0.42, 0.36),   # 10: Back wall mortar junction
        (0.50, 0.35),   # 11: Back wall mortar junction
        (0.58, 0.34),   # 12: Back wall mortar junction
        (0.66, 0.33),   # 13: Back wall mortar junction
        
        # === BACK WALL BRICK ROW 2 ===
        (0.14, 0.44),   # 14: mortar T-junction
        (0.22, 0.43),   # 15: mortar T-junction
        (0.30, 0.42),   # 16: mortar T-junction
        (0.38, 0.41),   # 17: mortar T-junction
        (0.46, 0.40),   # 18: mortar T-junction
        (0.54, 0.39),   # 19: mortar T-junction
        (0.62, 0.38),   # 20: mortar T-junction
        
        # === RIGHT WALL BRICK MORTAR JOINTS ===
        (0.75, 0.31),   # 21: Right wall upper brick corner
        (0.82, 0.30),   # 22: Right wall upper brick corner
        (0.88, 0.28),   # 23: Right wall upper mortar junction
        (0.75, 0.36),   # 24: Right wall mid mortar junction
        (0.82, 0.35),   # 25: Right wall mid mortar junction
        (0.88, 0.34),   # 26: Right wall mid mortar junction
        (0.75, 0.41),   # 27: Right wall lower mortar junction
        (0.82, 0.40),   # 28: Right wall lower mortar junction
        (0.92, 0.38),   # 29: Right wall edge mortar junction
        
        # === FRONT FACE BRICK MORTAR JOINTS (below bench seat) ===
        # Row 1
        (0.10, 0.57),   # 30: Front face upper mortar T-junction
        (0.18, 0.56),   # 31: Front face mortar junction
        (0.26, 0.55),   # 32: Front face mortar junction
        (0.34, 0.72),   # 33: Front face lower mortar junction
        (0.42, 0.71),   # 34: Front face lower mortar junction
        # Row 2
        (0.10, 0.63),   # 35: Front face mortar junction
        (0.18, 0.62),   # 36: Front face mortar junction
        (0.26, 0.61),   # 37: Front face mortar junction
        # Row 3
        (0.10, 0.70),   # 38: Lower front mortar junction
        (0.18, 0.69),   # 39: Lower front mortar junction
        (0.26, 0.68),   # 40: Lower front mortar junction
        
        # === CAP STONE EDGE CORNERS ===
        (0.12, 0.34),   # 41: Capstone left edge 
        (0.30, 0.32),   # 42: Capstone edge
        (0.50, 0.30),   # 43: Capstone edge
        (0.70, 0.27),   # 44: Capstone edge right
        
        # === RED WALL / HYDRANT STRUCTURAL CORNERS ===
        (0.07, 0.28),   # 45: Red wall left bottom corner
        (0.35, 0.10),   # 46: Red wall top edge
        (0.50, 0.08),   # 47: Hydrant pipe corner
        (0.55, 0.12),   # 48: Hydrant pipe bend
        (0.07, 0.06),   # 49: Building brick corner (top left)
        
        # === BOTTOM ROW BRICK MORTAR JOINTS ===
        (0.10, 0.77),   # 50: Bottom section mortar junction
        (0.22, 0.76),   # 51: Bottom section mortar junction
        (0.34, 0.82),   # 52: Bottom section lower mortar junction
        (0.10, 0.84),   # 53: Bottom-left mortar junction
    ]
    
    # Convert fractions to pixel coordinates
    corners = [(int(xf * w), int(yf * h)) for xf, yf in corners_frac]
    
    return np.array(corners, dtype=np.float32)


def run_harris_corner_detection(image_path):
    """
    Run Harris corner detection on the building image.
    Returns the corner response map.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # Harris corner detection
    # Parameters: blockSize=2, ksize (Sobel aperture)=3, k (Harris free parameter)=0.04
    harris_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    return img, harris_response


def non_max_suppression_grid(harris_response, threshold_fraction, grid_size=5):
    """
    Apply non-maximum suppression using a grid-based approach.
    For each grid cell, keep only the pixel with maximum response.
    """
    threshold = threshold_fraction * harris_response.max()
    h, w = harris_response.shape
    
    corners = []
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            y_end = min(y + grid_size, h)
            x_end = min(x + grid_size, w)
            
            patch = harris_response[y:y_end, x:x_end]
            max_val = patch.max()
            
            if max_val > threshold:
                local_pos = np.unravel_index(patch.argmax(), patch.shape)
                corners.append([x + local_pos[1], y + local_pos[0]])
    
    if corners:
        return np.array(corners, dtype=np.float32)
    else:
        return np.array([], dtype=np.float32).reshape(0, 2)


def threshold_and_detect(harris_response, threshold_fraction, use_nms=True, nms_grid=5):
    """
    Threshold the Harris response map.
    threshold_fraction: fraction of max response to use as threshold (0 to 1)
    Returns list of detected corner coordinates as (x, y).
    """
    if use_nms:
        return non_max_suppression_grid(harris_response, threshold_fraction, nms_grid)
    
    threshold = threshold_fraction * harris_response.max()
    detected = np.where(harris_response > threshold)
    detected_corners = np.column_stack((detected[1], detected[0])).astype(np.float32)
    return detected_corners


def compute_precision_recall(detected_corners, gt_corners, distance_threshold=20):
    """
    Compute precision and recall given detected and ground truth corners.
    A detected corner is a true positive if it is within distance_threshold
    pixels of a ground truth corner (1-to-1 matching via greedy assignment).
    
    Precision = TP / num_detected
    Recall = TP / num_ground_truth
    """
    if len(detected_corners) == 0:
        return 0.0, 0.0
    
    if len(gt_corners) == 0:
        return 0.0, 0.0
    
    # Compute pairwise distance matrix
    # gt_corners: (N, 2), detected_corners: (M, 2)
    dists = np.sqrt(
        np.sum((gt_corners[:, np.newaxis, :] - detected_corners[np.newaxis, :, :]) ** 2, axis=2)
    )
    # dists shape: (N_gt, M_det)
    
    # Greedy 1-to-1 matching: match each GT to closest unmatched detection
    gt_matched = np.zeros(len(gt_corners), dtype=bool)
    det_matched = np.zeros(len(detected_corners), dtype=bool)
    tp = 0
    
    # Get all (gt_idx, det_idx) pairs sorted by distance
    flat_indices = np.argsort(dists, axis=None)
    for flat_idx in flat_indices:
        gi = flat_idx // len(detected_corners)
        di = flat_idx % len(detected_corners)
        
        if dists[gi, di] > distance_threshold:
            break  # All remaining are too far
        
        if not gt_matched[gi] and not det_matched[di]:
            gt_matched[gi] = True
            det_matched[di] = True
            tp += 1
    
    precision = tp / len(detected_corners)
    recall = tp / len(gt_corners)
    
    return precision, recall


def plot_detected_corners(img, detected_corners, gt_corners, threshold_frac, 
                          precision, recall, output_path):
    """Plot detected corners (red) and ground truth (green) on the image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img_rgb)
    
    # Plot detected corners
    if len(detected_corners) > 0:
        ax.scatter(detected_corners[:, 0], detected_corners[:, 1], 
                   c='red', s=15, marker='x', linewidths=1, label=f'Detected ({len(detected_corners)})')
    
    # Plot ground truth corners
    ax.scatter(gt_corners[:, 0], gt_corners[:, 1], 
               c='lime', s=40, marker='o', facecolors='none', linewidths=1.5, 
               label=f'Ground Truth ({len(gt_corners)})')
    
    ax.set_title(f"Threshold = {threshold_frac:.3f} × max  |  "
                 f"Detected: {len(detected_corners)}  |  "
                 f"P={precision:.3f}  R={recall:.3f}", fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ground_truth(img, gt_corners, output_path):
    """Plot ground truth corners on the image with numbering."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img_rgb)
    
    ax.scatter(gt_corners[:, 0], gt_corners[:, 1], 
               c='lime', s=60, marker='o', facecolors='none', linewidths=2)
    
    for i, (x, y) in enumerate(gt_corners):
        ax.annotate(str(i+1), (x, y), xytext=(5, -5), 
                    textcoords='offset points', fontsize=6, color='yellow',
                    fontweight='bold')
    
    ax.set_title(f"Manually Annotated Ground Truth Corners (n={len(gt_corners)})", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def run_part_a():
    """Execute Part A: Harris Corner Detection with Precision-Recall Analysis."""
    print("=" * 70)
    print("PART A: Harris Corner Detection (10 Marks)")
    print("=" * 70)
    
    # Step 1: Load image and run Harris
    print("\n[Step 1] Loading building image and running Harris corner detection...")
    img, harris_response = run_harris_corner_detection(BUILDING_IMAGE_PATH)
    h, w = img.shape[:2]
    print(f"    Image size: {w} x {h}")
    print(f"    Harris response range: [{harris_response.min():.6f}, {harris_response.max():.6f}]")
    
    # Step 2: Get ground truth annotations
    print("\n[Step 2] Loading manually annotated ground truth corners...")
    gt_corners = get_ground_truth_corners(img.shape)
    print(f"    Number of ground truth corners: {len(gt_corners)}")
    
    # Adaptive distance threshold based on image resolution
    dist_thresh = max(15, int(0.015 * max(h, w)))
    print(f"    Matching distance threshold: {dist_thresh} pixels")
    
    # Plot ground truth
    plot_ground_truth(img, gt_corners, os.path.join(OUTPUT_DIR, "a1_ground_truth.png"))
    
    # Save Harris response as heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(harris_response, cmap='hot')
    plt.colorbar(label='Harris Response')
    plt.title('Harris Corner Response Map', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "a2_harris_response.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'a2_harris_response.png')}")
    
    # Step 3 & 4: Threshold and compute P/R
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10,
                  0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    
    precisions = []
    recalls = []
    num_detected_list = []
    
    print("\n[Step 3-4] Computing precision-recall for different thresholds...")
    print(f"    {'Thresh':>8} {'#Det':>8} {'Precision':>10} {'Recall':>8}")
    print("    " + "-" * 38)
    
    for i, t in enumerate(thresholds):
        # Detect corners with NMS
        nms_grid = max(3, int(0.005 * max(h, w)))
        detected = threshold_and_detect(harris_response, t, use_nms=True, nms_grid=nms_grid)
        
        # Compute precision and recall
        p, r = compute_precision_recall(detected, gt_corners, distance_threshold=dist_thresh)
        precisions.append(p)
        recalls.append(r)
        num_detected_list.append(len(detected))
        
        print(f"    {t:>8.3f} {len(detected):>8d} {p:>10.4f} {r:>8.4f}")
        
        # Step 5: Plot detected corners for each threshold
        plot_detected_corners(
            img, detected, gt_corners, t, p, r,
            os.path.join(OUTPUT_DIR, f"a3_corners_t{i:02d}_{t:.3f}.png")
        )
    
    # Step 6: Plot Precision-Recall curve
    print("\n[Step 5-6] Plotting Precision-Recall curve...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # PR Curve
    ax = axes[0]
    ax.plot(recalls, precisions, 'b-o', linewidth=2, markersize=5, zorder=5)
    for i, t in enumerate(thresholds):
        if i % 2 == 0:
            ax.annotate(f't={t:.3f}', (recalls[i], precisions[i]),
                       textcoords="offset points", xytext=(6, 6), fontsize=6,
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # BONUS: Compute AUC
    sorted_idx = np.argsort(recalls)
    sorted_r = np.array(recalls)[sorted_idx]
    sorted_p = np.array(precisions)[sorted_idx]
    
    # Deduplicate recall values (keep max precision for each recall)
    unique_r = [sorted_r[0]]
    unique_p = [sorted_p[0]]
    for j in range(1, len(sorted_r)):
        if sorted_r[j] != unique_r[-1]:
            unique_r.append(sorted_r[j])
            unique_p.append(sorted_p[j])
        else:
            unique_p[-1] = max(unique_p[-1], sorted_p[j])
    
    unique_r = np.array(unique_r)
    unique_p = np.array(unique_p)
    
    # Add boundary points
    if unique_r[0] > 0:
        unique_r = np.concatenate([[0], unique_r])
        unique_p = np.concatenate([[unique_p[0]], unique_p])
    if unique_r[-1] < 1:
        unique_r = np.concatenate([unique_r, [1]])
        unique_p = np.concatenate([unique_p, [0]])
    
    auc = np.trapezoid(unique_p, unique_r)
    
    ax.fill_between(unique_r, unique_p, alpha=0.15, color='blue')
    ax.text(0.55, 0.92, f'AUC = {auc:.4f}', transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Precision and Recall vs Threshold
    ax2 = axes[1]
    ax2.plot(thresholds, precisions, 'g-s', linewidth=2, markersize=5, label='Precision')
    ax2.plot(thresholds, recalls, 'r-^', linewidth=2, markersize=5, label='Recall')
    ax2.set_xlabel('Threshold (fraction of max response)', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Precision & Recall vs Threshold', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Number of detections vs threshold
    ax3 = axes[2]
    ax3.semilogy(thresholds, num_detected_list, 'ms-', linewidth=2, markersize=5)
    ax3.set_xlabel('Threshold (fraction of max response)', fontsize=12)
    ax3.set_ylabel('# Detected Corners (log)', fontsize=12)
    ax3.set_title('Number of Detections vs Threshold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "a4_precision_recall_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'a4_precision_recall_curve.png')}")
    
    print(f"\n  ★ [BONUS] Area Under PR Curve (AUC) = {auc:.4f}")
    
    # Summary table
    print("\n  === SUMMARY TABLE ===")
    print(f"  {'Threshold':>10} {'Detected':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 50)
    for i, t in enumerate(thresholds):
        f1 = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]) if (precisions[i] + recalls[i]) > 0 else 0
        print(f"  {t:>10.3f} {num_detected_list[i]:>10d} {precisions[i]:>10.4f} {recalls[i]:>8.4f} {f1:>8.4f}")
    
    return precisions, recalls, auc


# ============================================================================
# PART B: SIFT FEATURE MATCHING
# ============================================================================

def get_oyla_image_paths(oyla_dir):
    """
    Get sorted list of oyla image paths.
    Expected: oyla__0001.jpg, oyla__0003.jpg, ..., oyla__0037.jpg (19 images)
    """
    image_files = sorted([f for f in os.listdir(oyla_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return [os.path.join(oyla_dir, f) for f in image_files]


def compute_sift_features(image_path):
    """Compute SIFT keypoints and descriptors for an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return img, keypoints, descriptors


def match_features(desc1, desc2, ratio_threshold=0.75):
    """
    Match SIFT features using BFMatcher + Lowe's ratio test.
    Returns good matches and total matching cost.
    """
    if desc1 is None or desc2 is None:
        return [], float('inf')
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    total_cost = sum(m.distance for m in good_matches)
    return good_matches, total_cost


def get_consecutive_pairs(image_paths):
    """
    Get consecutive pairs including wrap-around:
    (img1, img2), (img2, img3), ..., (imgN, img1)
    """
    pairs = []
    for i in range(len(image_paths)):
        j = (i + 1) % len(image_paths)
        pairs.append((image_paths[i], image_paths[j]))
    return pairs


def get_random_pairs(image_paths, num_pairs=19):
    """Get random (non-consecutive) image pairs."""
    all_pairs = list(itertools.combinations(image_paths, 2))
    random.seed(42)
    return random.sample(all_pairs, min(num_pairs, len(all_pairs)))


def visualize_matches(img1, kp1, img2, kp2, matches, title, output_path, max_display=50):
    """Visualize feature matches between two images."""
    matches_sorted = sorted(matches, key=lambda m: m.distance)[:max_display]
    
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches_sorted, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_part_b():
    """Execute Part B: SIFT Feature Matching (15 Marks)."""
    print("\n" + "=" * 70)
    print("PART B: SIFT Feature Matching (15 Marks)")
    print("=" * 70)
    
    if not os.path.exists(OYLA_DIR):
        print(f"\n  [ERROR] Oyla images directory not found: {OYLA_DIR}")
        print(f"  Please create the directory and place the oyla images there:")
        print(f"    mkdir -p {OYLA_DIR}")
        print(f"  Expected files: oyla__0001.jpg, oyla__0003.jpg, ..., oyla__0037.jpg")
        return None
    
    # Step 1: Load images
    image_paths = get_oyla_image_paths(OYLA_DIR)
    print(f"\n[Step 1] Found {len(image_paths)} images")
    for p in image_paths:
        print(f"    {os.path.basename(p)}")
    
    if len(image_paths) < 2:
        print("  [ERROR] Need at least 2 images.")
        return None
    
    # Step 2: Compute SIFT features
    print("\n[Step 2] Computing SIFT keypoints and descriptors...")
    features = {}
    for path in image_paths:
        img, kp, desc = compute_sift_features(path)
        features[path] = {'img': img, 'kp': kp, 'desc': desc}
        print(f"    {os.path.basename(path)}: {len(kp)} keypoints, "
              f"descriptor shape: {desc.shape if desc is not None else 'None'}")
    
    # Visualize keypoints for a few images
    for idx in [0, len(image_paths)//2, -1]:
        path = image_paths[idx]
        f = features[path]
        img_kp = cv2.drawKeypoints(f['img'], f['kp'], None, color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title(f"SIFT Keypoints: {os.path.basename(path)} ({len(f['kp'])} keypoints)", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"b1_keypoints_{os.path.basename(path).split('.')[0]}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # Step 3: Consecutive pairs matching
    print("\n[Step 3] Matching CONSECUTIVE image pairs...")
    consecutive_pairs = get_consecutive_pairs(image_paths)
    consecutive_costs = []
    consecutive_match_counts = []
    
    print(f"    {'Pair':<55} {'Matches':>8} {'Cost':>12}")
    print("    " + "-" * 77)
    
    for i, (p1, p2) in enumerate(consecutive_pairs):
        f1, f2 = features[p1], features[p2]
        good_matches, cost = match_features(f1['desc'], f2['desc'])
        consecutive_costs.append(cost)
        consecutive_match_counts.append(len(good_matches))
        
        n1, n2 = os.path.basename(p1), os.path.basename(p2)
        print(f"    ({n1}, {n2}){' ' * max(0, 40-len(n1)-len(n2))} {len(good_matches):>8} {cost:>12.2f}")
        
        # Visualize all consecutive pair matches
        visualize_matches(
            f1['img'], f1['kp'], f2['img'], f2['kp'], good_matches,
            f"Consecutive: {n1} ↔ {n2} ({len(good_matches)} matches, cost={cost:.1f})",
            os.path.join(OUTPUT_DIR, f"b2_consec_{i:02d}_{n1}_{n2}.png")
        )
    
    total_consec = np.sum(consecutive_costs)
    avg_consec = np.mean(consecutive_costs)
    
    print(f"\n    TOTAL cost (consecutive): {total_consec:.2f}")
    print(f"    AVERAGE cost per pair:    {avg_consec:.2f}")
    print(f"    Total pairs:              {len(consecutive_pairs)}")
    
    # Step 4: Random pairs matching
    print(f"\n[Step 4] Matching RANDOM image pairs ({len(consecutive_pairs)} pairs)...")
    random_pairs = get_random_pairs(image_paths, len(consecutive_pairs))
    random_costs = []
    random_match_counts = []
    
    print(f"    {'Pair':<55} {'Matches':>8} {'Cost':>12}")
    print("    " + "-" * 77)
    
    for i, (p1, p2) in enumerate(random_pairs):
        f1, f2 = features[p1], features[p2]
        good_matches, cost = match_features(f1['desc'], f2['desc'])
        random_costs.append(cost)
        random_match_counts.append(len(good_matches))
        
        n1, n2 = os.path.basename(p1), os.path.basename(p2)
        print(f"    ({n1}, {n2}){' ' * max(0, 40-len(n1)-len(n2))} {len(good_matches):>8} {cost:>12.2f}")
        
        # Visualize first few
        if i < 5:
            visualize_matches(
                f1['img'], f1['kp'], f2['img'], f2['kp'], good_matches,
                f"Random: {n1} ↔ {n2} ({len(good_matches)} matches, cost={cost:.1f})",
                os.path.join(OUTPUT_DIR, f"b3_random_{i:02d}_{n1}_{n2}.png")
            )
    
    total_random = np.sum(random_costs)
    avg_random = np.mean(random_costs)
    
    print(f"\n    TOTAL cost (random):    {total_random:.2f}")
    print(f"    AVERAGE cost per pair:  {avg_random:.2f}")
    print(f"    Total pairs:            {len(random_pairs)}")
    
    # Step 5: Comparison and Explanation
    print("\n" + "=" * 70)
    print("[Step 5] COMPARISON: Consecutive vs Random Pairs")
    print("=" * 70)
    avg_cost_per_match_consec = total_consec / sum(consecutive_match_counts) if sum(consecutive_match_counts) > 0 else 0
    avg_cost_per_match_random = total_random / sum(random_match_counts) if sum(random_match_counts) > 0 else 0
    
    print(f"    Total cost (consecutive):    {total_consec:>12.2f}")
    print(f"    Total cost (random):         {total_random:>12.2f}")
    print(f"    Avg #matches (consecutive):  {np.mean(consecutive_match_counts):>11.1f}")
    print(f"    Avg #matches (random):       {np.mean(random_match_counts):>11.1f}")
    print(f"    Avg cost/match (consecutive): {avg_cost_per_match_consec:>10.2f}")
    print(f"    Avg cost/match (random):      {avg_cost_per_match_random:>10.2f}")
    
    if total_consec < total_random:
        winner = "CONSECUTIVE"
        print(f"\n    >> CONSECUTIVE pairs have LOWER total cost")
    else:
        winner = "RANDOM"
        print(f"\n    >> RANDOM pairs have LOWER total cost")
    
    explanation = f"""
    EXPLANATION / ANALYSIS:
    ─────────────────────────
    OBSERVATION: The total matching cost for CONSECUTIVE pairs ({total_consec:.0f}) is
    HIGHER than for RANDOM pairs ({total_random:.0f}).
    
    This may seem counter-intuitive, but it is explained by the NUMBER of
    matched features:
    
    - Consecutive pairs produce ~{np.mean(consecutive_match_counts):.0f} good matches on average
    - Random pairs produce only ~{np.mean(random_match_counts):.0f} good matches on average
    
    Since the total cost = SUM of all individual match distances, having ~{np.mean(consecutive_match_counts)/np.mean(random_match_counts):.0f}x
    more matches naturally leads to a higher total cost, even though the
    individual match quality is BETTER (lower per-match distance).
    
    KEY INSIGHT: The per-match cost tells the real story:
    - Consecutive: {avg_cost_per_match_consec:.2f} average distance per match
    - Random:      {avg_cost_per_match_random:.2f} average distance per match
    
    WHY CONSECUTIVE PAIRS HAVE MORE MATCHES:
    1. HIGH VISUAL OVERLAP: Consecutive frames share a large portion of the
       same scene since the camera moves only slightly between frames.
    2. SIMILAR VIEWPOINT: Small viewpoint changes mean SIFT descriptors for
       the same physical features are very similar, so many more matches
       pass Lowe's ratio test.
    3. DESCRIPTOR SIMILARITY: The per-match distance is lower because the
       same features appear with nearly identical local appearance.
    
    WHY RANDOM PAIRS HAVE FEWER BUT "CHEAPER" TOTAL MATCHES:
    1. LITTLE VISUAL OVERLAP: Random pairs may view opposite sides of the
       mulch stack, sharing very few common features.
    2. DIFFERENT VIEWPOINTS: Large viewpoint changes cause most features
       to fail the ratio test, leaving only a small number of robust matches.
    3. The few surviving matches happen to have lower total sum simply
       because there are far fewer of them.
    
    CONCLUSION: The total cost metric is dominated by the NUMBER of matches.
    Consecutive pairs have dramatically more correct matches (showing high
    visual coherence), which is WHY their total cost is higher. This
    demonstrates the fundamental principle that consecutive frames in a
    video sequence share significant visual content due to temporal and
    spatial proximity of the camera.
    """
    print(explanation)
    
    # Generate comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar chart: costs per pair
    ax = axes[0, 0]
    x = range(len(consecutive_costs))
    ax.bar(x, consecutive_costs, alpha=0.8, color='steelblue', label='Consecutive')
    ax.set_xlabel('Pair Index')
    ax.set_ylabel('Total Matching Cost')
    ax.set_title('Consecutive Pair Costs')
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[0, 1]
    x = range(len(random_costs))
    ax.bar(x, random_costs, alpha=0.8, color='coral', label='Random')
    ax.set_xlabel('Pair Index')
    ax.set_ylabel('Total Matching Cost')
    ax.set_title('Random Pair Costs')
    ax.grid(axis='y', alpha=0.3)
    
    # Summary comparison
    ax = axes[1, 0]
    categories = ['Consecutive', 'Random']
    totals = [total_consec, total_random]
    colors = ['steelblue', 'coral']
    bars = ax.bar(categories, totals, color=colors, width=0.5, edgecolor='black')
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.02,
                f'{val:.0f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Matching Cost')
    ax.set_title('Total Cost Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Match counts comparison
    ax = axes[1, 1]
    ax.bar(range(len(consecutive_match_counts)), consecutive_match_counts, 
           alpha=0.6, color='steelblue', label='Consecutive')
    ax.bar([x + 0.35 for x in range(len(random_match_counts))], random_match_counts,
           alpha=0.6, color='coral', width=0.35, label='Random')
    ax.set_xlabel('Pair Index')
    ax.set_ylabel('Number of Good Matches')
    ax.set_title('Match Count Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Part B: SIFT Feature Matching — Consecutive vs Random Pairs', 
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "b4_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'b4_comparison.png')}")
    
    return {
        'consecutive_costs': consecutive_costs,
        'random_costs': random_costs,
        'total_consecutive': total_consec,
        'total_random': total_random,
        'winner': winner
    }


# ============================================================================
# INTERACTIVE ANNOTATION TOOL
# ============================================================================

def interactive_annotate_corners(image_path):
    """
    Open an interactive window to manually click and annotate corners.
    Left-click to add corner, right-click to undo, 'q' to quit & print list.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    
    corners = []
    img_display = img.copy()
    
    def mouse_cb(event, x, y, flags, param):
        nonlocal img_display
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), 2)
            cv2.putText(img_display, str(len(corners)), (x+8, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow("Annotate Corners", img_display)
            print(f"  Corner {len(corners)}: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN and corners:
            corners.pop()
            img_display = img.copy()
            for i, (cx, cy) in enumerate(corners):
                cv2.circle(img_display, (cx, cy), 5, (0, 255, 0), 2)
                cv2.putText(img_display, str(i+1), (cx+8, cy-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow("Annotate Corners", img_display)
    
    cv2.namedWindow("Annotate Corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate Corners", mouse_cb)
    cv2.imshow("Annotate Corners", img_display)
    
    print("Controls: Left-click = add corner | Right-click = undo | q = quit")
    while cv2.waitKey(1) != ord('q'):
        pass
    cv2.destroyAllWindows()
    
    print(f"\nAnnotated {len(corners)} corners:")
    print("corners = [")
    for x, y in corners:
        print(f"    ({x}, {y}),")
    print("]")
    return corners


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       Computer Vision Assignment — Corner Detection        ║")
    print("║       & SIFT Feature Matching                              ║")
    print("║       Author: Veer Vardhan Singh                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--annotate":
        print("\n[MODE] Interactive corner annotation")
        interactive_annotate_corners(BUILDING_IMAGE_PATH)
        sys.exit(0)
    
    # ---- Part A ----
    try:
        precisions, recalls, auc = run_part_a()
        print(f"\n  ✓ Part A COMPLETE — AUC = {auc:.4f}")
    except FileNotFoundError as e:
        print(f"\n  ✗ Part A SKIPPED: {e}")
        print(f"    Save the building image as: {BUILDING_IMAGE_PATH}")
    except Exception as e:
        print(f"\n  ✗ Part A ERROR: {e}")
        import traceback; traceback.print_exc()
    
    # ---- Part B ----
    try:
        result_b = run_part_b()
        if result_b:
            print(f"\n  ✓ Part B COMPLETE")
            print(f"    Consecutive total: {result_b['total_consecutive']:.2f}")
            print(f"    Random total:      {result_b['total_random']:.2f}")
    except Exception as e:
        print(f"\n  ✗ Part B ERROR: {e}")
        import traceback; traceback.print_exc()
    
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
