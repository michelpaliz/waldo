import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🔍 Template Matching: Find All Waldos")

# Upload images
source_file = st.file_uploader("📸 Upload Full Scene Image", type=["png", "jpg", "jpeg"])
template_file = st.file_uploader("🔲 Upload Template Image", type=["png", "jpg", "jpeg"])

if source_file and template_file:
    # Convert to OpenCV format
    source_img = Image.open(source_file)
    template_img = Image.open(template_file)

    source_img_cv = np.array(source_img)
    template_img_cv = np.array(template_img)

    # Convert to grayscale
    source_gray = cv2.cvtColor(source_img_cv, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template_img_cv, cv2.COLOR_RGB2GRAY)

    # Ensure template is smaller
    if template_gray.shape[0] > source_gray.shape[0] or template_gray.shape[1] > source_gray.shape[1]:
        st.warning("⚠️ Template is larger than the source! Resizing it automatically.")
        scale_factor = 0.3
        new_width = int(template_gray.shape[1] * scale_factor)
        new_height = int(template_gray.shape[0] * scale_factor)
        template_gray = cv2.resize(template_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Display resized template
    st.image(template_gray, caption="🔍 Resized Template", use_column_width=True, clamp=True)

    # Perform template matching
    result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Lower the threshold to detect more matches
    threshold = 0.6  # Adjusted for better multiple detections
    y_locs, x_locs = np.where(result >= threshold)

    # Store detected bounding boxes
    h, w = template_gray.shape[:2]
    boxes = []
    
    for (x, y) in zip(x_locs, y_locs):
        boxes.append([x, y, x + w, y + h])

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    def nms(boxes, overlapThresh=0.3):
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        pick = []
        
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    # Apply Non-Maximum Suppression to clean up duplicate detections
    final_boxes = nms(boxes, overlapThresh=0.4)

    # Draw rectangles for each detected Waldo
    matched_img = source_img_cv.copy()
    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(matched_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Display results
    st.image(matched_img, caption="✅ All Detected Waldos (Green Boxes)", use_column_width=True)
