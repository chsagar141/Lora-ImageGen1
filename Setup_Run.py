import os
import cv2
import face_recognition # Assumed to be installed by setup_run.py

# Create folders if they don't exist
os.makedirs('auto-crop-face\\output', exist_ok=True)
os.makedirs('auto-crop-face\\failed', exist_ok=True)

def detect_faces(image_path):
    """
    Detects face locations in an image using the face_recognition library.

    Args:
        image_path (str): The path to the input image.

    Returns:
        list: A list of face locations (top, right, bottom, left) tuples.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        return face_locations
    except Exception as e:
        print(f"Error detecting faces in {image_path}: {e}")
        return []

def draw_rectangles_and_save(image_path, face_locations, output_path):
    """
    Crops an image based on face locations (or center if no faces)
    and saves it to a specified output path with 512x512 dimensions.

    Args:
        image_path (str): The path to the input image.
        face_locations (list): A list of face locations from detect_faces.
        output_path (str): The path where the processed image will be saved.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}. Skipping.")
        return

    height, width = image.shape[:2]
    orig_area = height * width
    cropped = None

    if face_locations:
        # Use the first detected face for cropping
        top, right, bottom, left = face_locations[0]

        face_width = right - left
        face_height = bottom - top

        # Add a margin around the face
        margin_x = int(face_width * 0.5)
        margin_y = int(face_height * 0.5)

        new_left = max(0, left - margin_x)
        new_top = max(0, top - margin_y)
        new_right = min(width, right + margin_x)
        new_bottom = min(height, bottom + margin_y)

        crop_width = new_right - new_left
        crop_height = new_bottom - new_top
        crop_area = crop_width * crop_height

        min_area = orig_area * 0.5 # Ensure cropped area is at least 50% of original

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

            # Adjust if cropping goes out of bounds to maintain target_side
            if new_right - new_left < target_side:
                new_left = max(0, new_right - target_side)
            if new_bottom - new_top < target_side:
                new_top = max(0, new_bottom - target_side)

        cropped = image[new_top:new_bottom, new_left:new_right]

    else:
        # If no faces detected, crop the central 50% of the image
        min_area = orig_area * 0.5
        target_side = int(min_area ** 0.5)
        target_side = min(target_side, height, width) # Ensure target_side doesn't exceed image dimensions

        center_y, center_x = height // 2, width // 2
        new_top = max(0, center_y - target_side // 2)
        new_left = max(0, center_x - target_side // 2)
        new_bottom = min(height, new_top + target_side)
        new_right = min(width, new_left + target_side)

        cropped = image[new_top:new_bottom, new_left:new_right]

    # Resize the cropped image to 512x512 and save
    if cropped is not None and cropped.shape[0] > 0 and cropped.shape[1] > 0:
        output_image = cv2.resize(cropped, (512, 512))
        cv2.imwrite(output_path, output_image)
    else:
        print(f"Warning: Cropped image for {image_path} is empty or invalid. Not saving.")


def main():
    """
    Main function to process images in the 'load' folder, detect faces,
    crop/resize, and save to 'output' or 'failed' folders.
    """
    load_folder = 'auto-crop-face\\load'
    output_folder = 'auto-crop-face\\output'
    failed_folder = 'auto-crop-face\\failed'

    # Ensure load folder exists, though it's expected to be populated by user
    os.makedirs(load_folder, exist_ok=True)

    print(f"\n--- Starting Face Detection and Cropping (run.py) ---")
    print(f"Looking for images in: {load_folder}")

    images_found = False
    for filename in os.listdir(load_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            images_found = True
            image_path = os.path.join(load_folder, filename)
            print(f"Processing: {filename}")

            face_locations = detect_faces(image_path)

            if face_locations:
                output_path = os.path.join(output_folder, filename)
                print(f"  ✅ Faces detected in {filename} ({len(face_locations)} face(s))")
            else:
                output_path = os.path.join(failed_folder, filename)
                print(f"  ❌ No faces detected in {filename}")

            draw_rectangles_and_save(image_path, face_locations, output_path)

    if not images_found:
        print(f"No image files found in '{load_folder}'. Please place images there to process.")
    print("--- Face Detection and Cropping (run.py) Complete ---")

if __name__ == "__main__":
    main()
