import cv2 as cv
import pandas as pd
import os

# Paths
dataset_path = "test"  # Replace with your dataset path
annotations_file = "_annotations.csv"  # Replace with your annotations CSV file
output_path = "images"
os.makedirs(output_path, exist_ok=True)

# Load annotations
annotations = pd.read_csv(annotations_file)

# Group annotations by filename for processing
grouped_annotations = annotations.groupby('filename')

# Process each image
for filename, group in grouped_annotations:
    image_path = os.path.join(dataset_path, filename)
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # Load as grayscale
    if image is None:
        print(f"Image {filename} not found or could not be loaded.")
        continue

    # Convert to BGR for annotation
    output_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # Process iris and pupil annotations
    iris_center, iris_radius = None, None
    pupil_center, pupil_radius = None, None

    for _, row in group.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        radius = max((xmax - xmin), (ymax - ymin)) // 2  # Approximate radius as half of the bounding box width or height

        if row['class'] == 'iris':
            iris_center = (center_x, center_y)
            iris_radius = radius
        elif row['class'] == 'pupille':
            pupil_center = (center_x, center_y)
            pupil_radius = radius

    # Draw circles and annotate if both iris and pupil are detected
    if iris_center and iris_radius:
        cv.circle(output_image, iris_center, iris_radius, (255, 0, 255), 2)  # Purple for iris
        iris_diameter = 2 * iris_radius

    if pupil_center and pupil_radius:
        cv.circle(output_image, pupil_center, pupil_radius, (0, 255, 0), 2)  # Green for pupil
        pupil_diameter = 2 * pupil_radius

    # Calculate and display ratio if both are present
    if iris_center and pupil_center and iris_radius and pupil_radius:
        ratio = iris_radius / pupil_radius if pupil_radius != 0 else 0

        # Annotate diameters and ratio on the image
        cv.putText(output_image, f"Iris Diameter: {iris_diameter:.2f} px", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv.putText(output_image, f"Pupil Diameter: {pupil_diameter:.2f} px", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(output_image, f"Ratio: {ratio:.2f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Save the annotated image
    output_file = os.path.join(output_path, f"annotated_{filename}")
    cv.imwrite(output_file, output_image)
    print(f"Processed and saved: {output_file}")
