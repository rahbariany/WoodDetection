import json
import numpy as np
import cv2
import os

def create_grouped_masks_from_annotations(json_file, output_dir, image_size):
    """
    Generate grouped black-and-white mask images from a CVAT JSON file in a custom format.

    Args:
        json_file (str): Path to the input JSON file.
        output_dir (str): Directory to save the grouped mask images.
        image_size (tuple): Size of the output mask (width, height).
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to hold masks grouped by frame
    grouped_masks = {}

    # Iterate through the top-level list
    for item in data:
        shapes = item.get("shapes", [])

        # Iterate through the shapes
        for shape in shapes:
            frame = shape.get("frame", 0)  # Group by frame index
            points = shape.get("points", [])
            shape_type = shape.get("type", "polygon")

            # Initialize a mask for the frame if not already created
            if frame not in grouped_masks:
                grouped_masks[frame] = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

            # Validate points
            if len(points) % 2 == 0 and len(points) > 0:
                points_array = np.array(points, dtype=np.int32).reshape((-1, 2))

                # Draw the shapes based on their type
                if shape_type == "polygon":
                    if points_array.shape[0] >= 3:  # Minimum of 3 points for a valid polygon
                        cv2.fillPoly(grouped_masks[frame], [points_array], 255)
                    else:
                        print(f"Skipping polygon with insufficient points: {points}")
                elif shape_type == "polyline":
                    if points_array.shape[0] >= 2:  # Minimum of 2 points for a polyline
                        cv2.polylines(grouped_masks[frame], [points_array], isClosed=False, color=255, thickness=2)
                    else:
                        print(f"Skipping polyline with insufficient points: {points}")
                elif shape_type == "ellipse":
                    if len(points) >= 4:
                        center = tuple(map(int, points[:2]))
                        axes = tuple(map(int, points[2:4]))
                        angle = int(points[4]) if len(points) > 4 else 0
                        cv2.ellipse(grouped_masks[frame], center, axes, angle, 0, 360, 255, -1)
                    else:
                        print(f"Skipping ellipse with insufficient points: {points}")
            else:
                print(f"Invalid or empty points array: {points}")

    # Save the grouped mask images
    for frame, mask in grouped_masks.items():
        output_file = os.path.join(output_dir, f"mask_frame_{frame}_1500x1500.png")
        cv2.imwrite(output_file, mask)

# Example usage
input_json = "/Users/mohammadmahdi/Downloads/backup_cvat_22.01.2025/task_38/annotations.json"  # Replace with the correct path to your JSON file
output_directory = "/Users/mohammadmahdi/Downloads/backup_cvat_22.01.2025/task_38/grouped_masks"  # Replace with your desired output directory
image_dimensions = (1500, 1500)  # Replace with your image resolution (width, height)

create_grouped_masks_from_annotations(input_json, output_directory, image_dimensions)
