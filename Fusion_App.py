import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

def capture_images(num_images=10):
    st.info(f"Capturing {num_images} images. Please wait...")
    cap = cv2.VideoCapture(0)
    images = []
    for i in range(num_images):
        ret, frame = cap.read()
        if ret:
            images.append(frame)
        else:
            st.error("Failed to capture image")
    cap.release()
    return images

def align_images(images):
    ref_image = images[0]
    aligned_images = [ref_image]
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref_image, None)
    
    for img in images[1:]:
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        aligned_img = cv2.warpPerspective(img, M, (ref_image.shape[1], ref_image.shape[0]))
        aligned_images.append(aligned_img)
    return aligned_images

def average_fusion(images):
    aligned_images = align_images(images)
    fused_image = np.mean(aligned_images, axis=0).astype(np.uint8)
    return fused_image

st.title("Image Fusion App")

if st.button("Capture Images"):
    images = capture_images(num_images=10)
    if images:
        st.success("Images captured successfully")
        st.subheader("Captured Images:")
        for i, img in enumerate(images):
            st.image(img, channels="BGR", caption=f"Image {i+1}")
        
        fused_image = average_fusion(images)
        st.subheader("Fused Image:")
        st.image(fused_image, channels="BGR", caption="Fused Image")
        
        result = Image.fromarray(cv2.cvtColor(fused_image, cv2.COLOR_BGR2RGB))
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            result.save(temp_file.name)
            st.markdown(f"Download the fused image: [Download Image]({temp_file.name})")
