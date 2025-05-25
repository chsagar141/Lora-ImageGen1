import os
import cv2
import face_recognition
import logging
import numpy as np
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Cross-platform folder paths
base_dir = 'auto-crop-face'
load_folder = os.path.join(base_dir, 'load')
output_folder = os.path.join(base_dir, 'output')
failed_folder = os.path.join(base_dir, 'failed')

# Ensure folders exist
os.makedirs(load_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(failed_folder, exist_ok=True)

def detect_faces(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        return face_locations
    except Exception as e:
        logging.error(f"Error detecting faces in {image_path}: {e}")
        return []

def draw_rectangles_and_save(image_path, face_locations, output_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Could not read image {image_path}. Skipping.")
        return False

    height, width = image.shape[:2]
    if height < 100 or width < 100:
        logging.warning(f"Image {image_path} is too small ({width}x{height}). Skipping.")
        return False

    orig_area = height * width
    cropped = None

    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top
        margin_x = int(face_width * 0.5)
        margin_y = int(face_height * 0.5)

        new_left = max(0, left - margin_x)
        new_top = max(0, top - margin_y)
        new_right = min(width, right + margin_x)
        new_bottom = min(height, bottom + margin_y)

        crop_width = new_right - new_left
        crop_height = new_bottom - new_top
        crop_area = crop_width * crop_height
        min_area = orig_area * 0.5

        if crop_area < min_area:
            target_area = min_area
            target_side = int(target_area ** 0.5)
            center_x = new_left + crop_width // 2
            center_y = new_top + crop_height // 2
            half_side = target_side // 2

            new_left = max(0, center_x - half_side)
            new_top = max(0, center_y - half_side)
            new_right = min(width, center_x + half_side)
            new_bottom = min(height, center_y + half_side)

            if new_right - new_left < target_side:
                new_left = max(0, new_right - target_side)
            if new_bottom - new_top < target_side:
                new_top = max(0, new_bottom - target_side)

        cropped = image[new_top:new_bottom, new_left:new_right]
    else:
        min_area = orig_area * 0.5
        target_side = int(min(min_area ** 0.5, height, width))
        center_y, center_x = height // 2, width // 2
        new_top = max(0, center_y - target_side // 2)
        new_left = max(0, center_x - target_side // 2)
        new_bottom = min(height, new_top + target_side)
        new_right = min(width, new_left + target_side)

        cropped = image[new_top:new_bottom, new_left:new_right]

    if cropped is not None and cropped.shape[0] > 0 and cropped.shape[1] > 0:
        # Pad if too small before resizing
        h, w = cropped.shape[:2]
        if h < 512 or w < 512:
            pad_y = max(0, (512 - h) // 2)
            pad_x = max(0, (512 - w) // 2)
            cropped = cv2.copyMakeBorder(cropped, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)

        output_image = cv2.resize(cropped, (512, 512))
        cv2.imwrite(output_path, output_image)
        return True
    else:
        logging.warning(f"Cropped image for {image_path} is empty or invalid.")
        return False

def run_yolo_if_needed():
    failed_images = [f for f in os.listdir(failed_folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if failed_images:
        logging.info(f"⚠️ {len(failed_images)} image(s) in 'failed' folder. Running yolo.py...")
        try:
            subprocess.run(["python", "auto-crop-face\yolo.py"], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running yolo.py: {e}")

def main():
    logging.info("--- Starting Face Detection and Cropping (run.py) ---")
    logging.info(f"Looking for images in: {load_folder}")

    images_found = False
    for filename in os.listdir(load_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            images_found = True
            image_path = os.path.join(load_folder, filename)
            logging.info(f"Processing: {filename}")

            face_locations = detect_faces(image_path)

            if face_locations:
                logging.info(f"  ✅ Faces detected ({len(face_locations)} face(s))")
                output_path = os.path.join(output_folder, filename)
            else:
                logging.info(f"  ❌ No faces detected")
                output_path = os.path.join(failed_folder, filename)

            success = draw_rectangles_and_save(image_path, face_locations, output_path)
            if not success:
                logging.warning(f"  ⚠️ Failed to save: {filename}")

    if not images_found:
        logging.info(f"No image files found in '{load_folder}'. Please place images there to process.")

    # Trigger YOLO fallback if needed
    run_yolo_if_needed()

    logging.info("--- Face Detection and Cropping Complete ---")

if __name__ == "__main__":
    main()
