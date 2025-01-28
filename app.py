import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🔍 Template Matching: Find Waldo (Detect Multiple Instances)")

# Upload the full scene
source_file = st.file_uploader("📸 Upload Full Scene Image", type=["png", "jpg", "jpeg"])
# Upload the cropped template
template_file = st.file_uploader("🔲 Upload Template Image", type=["png", "jpg", "jpeg"])

if source_file and template_file:
    # Convert to OpenCV format
    source_img = Image.open(source_file)
    template_img = Image.open(template_file)
    
    source_img_cv = np.array(source_img)
    template_img_cv = np.array(template_img)

    # Convert images to grayscale
    source_gray = cv2.cvtColor(source_img_cv, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template_img_cv, cv2.COLOR_RGB2GRAY)

    # Ensure the template is smaller than the source
    if template_gray.shape[0] > source_gray.shape[0] or template_gray.shape[1] > source_gray.shape[1]:
        st.warning("⚠️ The template is larger than the source image. Resizing the template automatically.")
        scale_factor = 0.3  # Reduce template size (30% of original size)
        new_width = int(template_gray.shape[1] * scale_factor)
        new_height = int(template_gray.shape[0] * scale_factor)
        template_gray = cv2.resize(template_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Display resized template for reference
    st.image(template_gray, caption="🔍 Resized Template", use_column_width=True, clamp=True)

    # Perform template matching
    result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Define a similarity threshold (adjust as needed)
    threshold = 0.8  # Matches with similarity above this value will be detected
    y_locs, x_locs = np.where(result >= threshold)  # Find all positions above the threshold

    # Draw rectangles around all detected matches
    matched_img = source_img_cv.copy()
    h, w = template_gray.shape[:2]
    for (x, y) in zip(x_locs, y_locs):
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(matched_img, top_left, bottom_right, (0, 255, 0), 3)  # Green rectangles

    # Display results
    st.image(matched_img, caption="✅ Detected Objects (Green Boxes)", use_column_width=True)
