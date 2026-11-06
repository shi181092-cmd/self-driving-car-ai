import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="AI Self-Driving Car Simulator", page_icon="🚗", layout="wide")

st.title("🚗 AI Self-Driving Car Simulator")
st.caption("A learning prototype that detects objects in road images/videos and suggests a simulated driving action.")

st.warning(
    "Educational demo only. This is NOT a real autonomous-driving system. "
    "Do not connect it to a real vehicle or use its output for driving decisions."
)

@st.cache_resource
def load_model():
    # Downloads the small pretrained YOLOv8 model the first time the app runs.
    return YOLO("yolov8n.pt")

def classify_traffic_light(crop):
    """Very simple colour heuristic; not reliable enough for real driving."""
    if crop is None or crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = (s > 90) & (v > 90)
    red = (((h <= 10) | (h >= 170)) & valid).sum()
    yellow = ((h >= 15) & (h <= 40) & valid).sum()
    green = ((h >= 40) & (h <= 90) & valid).sum()
    counts = {"red": int(red), "yellow": int(yellow), "green": int(green)}
    colour, amount = max(counts.items(), key=lambda item: item[1])
    if amount < 5:
        return "unknown"
    return colour

def analyse_frame(frame_bgr, model, confidence):
    results = model.predict(frame_bgr, conf=confidence, verbose=False)
    result = results[0]
    names = model.names
    detections = []
    annotated = frame_bgr.copy()
    traffic_colours = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2); y2 = min(frame_bgr.shape[0], y2)

            extra = ""
            if label == "traffic light" and x2 > x1 and y2 > y1:
                extra = classify_traffic_light(frame_bgr[y1:y2, x1:x2])
                traffic_colours.append(extra)

            display_label = f"{label} {score:.2f}"
            if extra:
                display_label += f" ({extra})"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 80), 2)
            cv2.putText(annotated, display_label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 80), 2)

            area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame_bgr.shape[0] * frame_bgr.shape[1])
            detections.append({
                "object": label,
                "confidence": round(score, 3),
                "area_ratio": area_ratio,
                "traffic_colour": extra
            })

    action, reason = decide_action(detections, traffic_colours)
    return annotated, detections, action, reason

def decide_action(detections, traffic_colours):
    labels = [d["object"] for d in detections]

    if "person" in labels:
        return "STOP", "A pedestrian/person was detected. The simulator uses a cautious stop rule."
    if "stop sign" in labels:
        return "STOP", "A stop sign was detected."
    if "red" in traffic_colours:
        return "STOP", "A traffic-light crop was heuristically classified as red."
    if "yellow" in traffic_colours:
        return "SLOW DOWN", "A traffic-light crop was heuristically classified as yellow."

    nearby_vehicle_labels = {"car", "bus", "truck", "motorcycle", "bicycle"}
    large_vehicle = any(
        d["object"] in nearby_vehicle_labels and d["area_ratio"] > 0.18
        for d in detections
    )
    if large_vehicle:
        return "SLOW DOWN", "A detected vehicle occupies a large part of the image; this is only a rough visual heuristic."

    if not detections:
        return "CAUTION", "No supported objects were detected. The system cannot confirm that the road is clear."
    return "MOVE FORWARD (SIMULATED)", "No stop condition was triggered by the simple demonstration rules."

def show_results(frame_bgr, model, confidence):
    annotated, detections, action, reason = analyse_frame(frame_bgr, model, confidence)
    left, right = st.columns([3, 2])
    with left:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="AI object-detection output", use_container_width=True)
    with right:
        st.subheader("Simulated driving decision")
        if action == "STOP":
            st.error(f"🛑 {action}")
        elif action == "SLOW DOWN":
            st.warning(f"⚠️ {action}")
        elif action.startswith("MOVE"):
            st.success(f"➡️ {action}")
        else:
            st.info(f"❔ {action}")
        st.write(reason)
        st.metric("Objects detected", len(detections))
        if detections:
            st.dataframe(
                [{"Object": d["object"], "Confidence": d["confidence"],
                  "Traffic-light colour": d["traffic_colour"] or "—"} for d in detections],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.write("No objects from the model's supported classes were detected.")

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    st.markdown("---")
    st.write("Model: YOLOv8 Nano (pretrained COCO weights)")
    st.write("The first run downloads the model weights and may take a few minutes.")

tab_image, tab_video, tab_info = st.tabs(["📷 Image demo", "🎞️ Video demo", "ℹ️ How it works"])

with tab_image:
    st.subheader("Upload a road image")
    image_file = st.file_uploader("Choose JPG, JPEG or PNG", type=["jpg", "jpeg", "png"], key="image")
    if image_file is not None:
        pil_image = Image.open(image_file).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if st.button("Analyse image", type="primary"):
            with st.spinner("Loading AI model and analysing image..."):
                model = load_model()
                show_results(frame, model, confidence)

with tab_video:
    st.subheader("Upload a short road video")
    st.caption("For a quick demonstration, use a short MP4 video. Processing every frame can take time.")
    video_file = st.file_uploader("Choose MP4, MOV or AVI", type=["mp4", "mov", "avi"], key="video")
    max_frames = st.slider("Maximum frames to analyse", 5, 60, 20, 5)
    if video_file is not None and st.button("Analyse video", type="primary"):
        suffix = os.path.splitext(video_file.name)[1] or ".mp4"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(video_file.read())
                temp_path = temp.name
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("Could not open this video. Try another MP4 file.")
            else:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // max_frames) if total > 0 else 1
                model = load_model()
                frame_number = 0
                analysed = 0
                progress = st.progress(0)
                output_area = st.container()
                while analysed < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_number % step == 0:
                        annotated, detections, action, reason = analyse_frame(frame, model, confidence)
                        with output_area:
                            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                     caption=f"Frame {frame_number}: {action} — {reason}",
                                     use_container_width=True)
                        analysed += 1
                        progress.progress(min(1.0, analysed / max_frames))
                    frame_number += 1
                cap.release()
                if analysed == 0:
                    st.error("No frames could be read from this video.")
                else:
                    st.success(f"Finished analysing {analysed} frames.")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

with tab_info:
    st.subheader("Project workflow")
    st.markdown("""
    1. **Input:** The user uploads a road image or video.
    2. **Perception:** YOLO detects supported objects such as people, cars, buses, trucks, bicycles, traffic lights and stop signs.
    3. **Traffic-light colour estimate:** A basic OpenCV colour heuristic estimates red, yellow or green inside a detected traffic-light box.
    4. **Decision:** Simple rules map detections to STOP, SLOW DOWN, CAUTION or MOVE FORWARD (SIMULATED).
    5. **Output:** The app displays annotated images, object labels and the suggested action.

    **Important limitations:** This prototype does not estimate true distance, understand right-of-way, plan a safe route, or control a vehicle. The traffic-light colour heuristic and decision rules can make mistakes. A real autonomous vehicle needs extensive sensor fusion, localization, prediction, planning, fail-safe systems and safety validation.
    """)
    st.subheader("YOLO classes used")
    st.write("Person, car, bus, truck, motorcycle, bicycle, traffic light and stop sign (among other COCO classes).")
