import cv2
import numpy as np
import os


def compute_mask_accuracy(cvat_mask_path, test_mask_path):
    """
    Compute the accuracy and IoU of a test mask compared to a reference CVAT mask.

    Args:
        cvat_mask_path (str): Path to the CVAT-generated mask.
        test_mask_path (str): Path to the test mask to evaluate.

    Returns:
        dict: Dictionary containing IoU and accuracy values.
    """
    # Load the masks as grayscale images
    cvat_mask = cv2.imread(cvat_mask_path, cv2.IMREAD_GRAYSCALE)
    test_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both masks are loaded correctly
    if cvat_mask is None or test_mask is None:
        print(f"Error: Could not load one of the masks - {cvat_mask_path} or {test_mask_path}")
        return {"IoU": 0.0, "Accuracy": 0.0}

    # Ensure the masks have the same size
    if cvat_mask.shape != test_mask.shape:
        test_mask = cv2.resize(test_mask, (cvat_mask.shape[1], cvat_mask.shape[0]))

    # Convert masks to binary (0 or 255)
    _, cvat_mask = cv2.threshold(cvat_mask, 127, 255, cv2.THRESH_BINARY)
    _, test_mask = cv2.threshold(test_mask, 127, 255, cv2.THRESH_BINARY)

    # Compute intersection and union
    intersection = np.logical_and(cvat_mask, test_mask).sum()
    union = np.logical_or(cvat_mask, test_mask).sum()

    # Compute IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0.0

    # Compute pixel accuracy
    total_pixels = cvat_mask.shape[0] * cvat_mask.shape[1]
    correct_pixels = np.sum(cvat_mask == test_mask)
    accuracy = correct_pixels / total_pixels

    return {"IoU": iou, "Accuracy": accuracy}


def evaluate_multiple_masks(cvat_mask_dir, test_mask_dir):
    """
    Compare multiple test masks against CVAT reference masks and compute overall accuracy.

    Args:
        cvat_mask_dir (str): Directory containing CVAT-generated masks.
        test_mask_dir (str): Directory containing test masks to evaluate.

    Returns:
        dict: Dictionary containing individual and overall IoU and accuracy.
    """
    results = {}
    total_iou = 0.0
    total_accuracy = 0.0
    count = 0

    for filename in sorted(os.listdir(cvat_mask_dir)):
        if filename.startswith("mask_frame") and filename.endswith(".png"):  # Ensure only mask files are processed
            frame_number = filename.split("_")[2].split(".")[0]  # Extract frame number correctly
            test_mask_filename = f"mejorada_recortada_0301-{int(frame_number) + 1}_mascara.png"
            cvat_mask_path = os.path.join(cvat_mask_dir, filename)
            test_mask_path = os.path.join(test_mask_dir, test_mask_filename)

            if os.path.exists(test_mask_path):
                metrics = compute_mask_accuracy(cvat_mask_path, test_mask_path)
                results[filename] = metrics
                total_iou += metrics["IoU"]
                total_accuracy += metrics["Accuracy"]
                count += 1
            else:
                print(f"Warning: Missing test mask for {filename} -> Expected: {test_mask_filename}")

    # Compute overall IoU and accuracy
    overall_iou = total_iou / count if count > 0 else 0.0
    overall_accuracy = total_accuracy / count if count > 0 else 0.0

    results["Overall"] = {"IoU": overall_iou, "Accuracy": overall_accuracy}
    return results


# Example usage
cvat_masks_folder = "/Users/mohammadmahdi/Downloads/backup_cvat_22.01.2025.withmasks/task_33/grouped_masks"  # Path to CVAT-generated masks
test_masks_folder = "/Users/mohammadmahdi/Downloads/prediction_sam"  # Path to test masks

comparison_results = evaluate_multiple_masks(cvat_masks_folder, test_masks_folder)

# Print results
# Print results with explicit mask comparisons
for mask_name, metrics in comparison_results.items():
    if mask_name != "Overall":  # Skip the overall entry in results
        # Extract frame number from the mask_name to construct the corresponding test mask filename
        frame_number = mask_name.split("_")[2].split(".")[0]  # Extract frame number
        test_mask_filename = f"mejorada_recortada_0301-{int(frame_number) + 1}_mascara.png"

        print(f"Comparing {mask_name} (CVAT mask) with {test_mask_filename} (Test mask):")
        print(f"    IoU = {metrics['IoU']:.4f}, Accuracy = {metrics['Accuracy']:.4f}")
    else:
        print(f"Overall comparison: IoU = {metrics['IoU']:.4f}, Accuracy = {metrics['Accuracy']:.4f}")