import cv2
import numpy as np
import os

# --- SETTINGS ---
image_path = "..\data\Datathon Mosquito Species Images\Aedes aegypti\JPEG\IMG_7293.jpg"  # your input image
output_folder = "..\data\crops"  # folder to save cropped images

# --- PREP OUTPUT FOLDER ---
os.makedirs(output_folder, exist_ok=True)

# --- LOAD IMAGE ---
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- THRESHOLD ---
# Adjust 250 depending on brightness (lower = detect more)
_, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

# --- FIND CONTOURS ---
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours")

# --- LOOP THROUGH CONTOURS AND SAVE EACH CROPPED BOX ---
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    # Skip tiny specks (noise)
    if w < 100 or h < 100 or w > 400 or h > 400:
        continue
    """
    # --- NEW CODE FOR 1:1 SQUARE CROP ---
    # 1. Find the maximum dimension
    max_dim = max(w, h)

    # 2. Calculate the difference (extra space) needed for centering
    # This extra space will be distributed equally on both sides
    x_extra = max_dim - w
    y_extra = max_dim - h

    # 3. Adjust the coordinates to create a square centered on the original box
    # Subtract half the extra space from the start coordinate (x, y)
    # The new width and height will both be max_dim
    x1 = x - (x_extra // 2)
    y1 = y - (y_extra // 2)

    # 4. Make sure the new coordinates don't go out of the image bounds
    # (This is a minimal bounding check; a complete solution needs image dimensions)
    # For a robust solution, you should define image_h and image_w outside the loop.
    # For now, we'll keep the direct cropping line as requested:

    # 5. The modified cropping line:
    crop = image[y1: y1 + max_dim, x1: x1 + max_dim]
"""






    crop = image[y:y + h, x:x + w]
    output_path = os.path.join(output_folder, f"crop_{i + 1}.jpg")
    cv2.imwrite(output_path, crop)
    print(f"Saved {output_path}")

print("✅ Done! All cropped regions saved in:", output_folder)

'''





import cv2
import numpy as np
import os

# --- SETTINGS ---
image_path = "..\data\Datathon Mosquito Species Images\Aedes aegypti\JPEG\IMG_7293.jpg"  # your input image
output_folder = "..\data\crops"  # folder to save cropped images

# --- PREP OUTPUT FOLDER ---
os.makedirs(output_folder, exist_ok=True)

# --- LOAD IMAGE ---
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Error: Could not load image from {image_path}")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- THRESHOLD ---
# Adjust 250 depending on brightness (lower = detect more)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)


# --- FIND CONTOURS ---
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours")

# Get image dimensions for boundary checks
img_h, img_w = image.shape[:2]

# --- LOOP THROUGH CONTOURS AND SAVE EACH CROPPED BOX (1:1 RATIO) ---
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    # Skip tiny specks (noise)
    if w < 50 or h < 50:
        continue

    # --- MAKE THE BOX SQUARE (1:1 RATIO) ---
    # Find the maximum dimension (either width or height)
    max_dim = max(w, h)

    # Calculate the new square dimensions
    # New width and height will both be max_dim
    new_w = max_dim
    new_h = max_dim

    # Calculate the extra space needed for the new dimension
    # This centers the original contour within the new square box
    x_center_offset = (new_w - w) // 2
    y_center_offset = (new_h - h) // 2

    # Calculate the new top-left coordinates for the square box
    x1 = x - x_center_offset
    y1 = y - y_center_offset

    # Calculate the new bottom-right coordinates
    x2 = x1 + new_w
    y2 = y1 + new_h

    # --- ADJUST FOR IMAGE BOUNDARIES ---
    # Clamp coordinates to stay within the image borders
    # This prevents the crop from going out of bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Recalculate the final width and height after clamping,
    # ensuring it's a square based on the smallest dimension
    final_w = x2 - x1
    final_h = y2 - y1

    # Take the smaller of the two dimensions to create the final square
    # This handles cases where clamping against the boundary breaks the square
    final_dim = min(final_w, final_h)

    # Adjust the crop region to be a perfect square using the 'final_dim'
    # Prioritize cropping from the top-left (x1, y1) for simplicity
    x_crop_end = x1 + final_dim
    y_crop_end = y1 + final_dim


    # --- PERFORM CROP AND SAVE ---
    crop = image[y1:y_crop_end, x1:x_crop_end]
    output_path = os.path.join(output_folder, f"crop_{i + 1}.jpg")
    cv2.imwrite(output_path, crop)
    print(f"Saved {output_path} (Size: {crop.shape[1]}x{crop.shape[0]})")

print("✅ Done! All cropped regions saved in:", output_folder)

'''
