"""
Streamlit Web Application for Real-Time Object Detection and Tracking.
"""

import tempfile
import time
from collections import defaultdict, deque

import cv2
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from core.detector import Detector
from core.tracker import CentroidTracker
from core.utils import calculate_fps, draw_detections, draw_fps, draw_stats

# 2. PAGE CONFIG
st.set_page_config(page_title="Object Detection & Tracking", layout="wide", page_icon="🎯")

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 3. SIDEBAR
st.sidebar.title("Object Detection & Tracking")
st.sidebar.markdown("**Living = Red | Non-living = Gray**")

conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
source_choice = st.sidebar.radio("Input source", options=["Webcam (live)", "Upload a video file"])

uploaded_file = None
if source_choice == "Upload a video file":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

allowed_classes = st.sidebar.multiselect("Filter classes to detect", options=COCO_CLASSES, default=COCO_CLASSES)

show_trails = st.sidebar.toggle("Show trajectory trails", value=True)
show_stats = st.sidebar.toggle("Show stats panel", value=True)

start_btn = st.sidebar.button("Start Detection")
stop_btn = st.sidebar.button("Stop")

# 4. MAIN PANEL
col1, col2 = st.columns([2, 1])

with col1:
    video_placeholder = st.empty()

with col2:
    st.markdown("### Live Metrics")
    metric_total = st.empty()
    metric_living = st.empty()
    metric_nonliving = st.empty()
    metric_fps = st.empty()
    
    st.markdown("### Session Log")
    log_placeholder = st.empty()
    
    st.markdown("### Live Charts")
    chart1_placeholder = st.empty()
    chart2_placeholder = st.empty()
    chart3_placeholder = st.empty()

# 5. DETECTION LOOP
if start_btn:
    if source_choice == "Webcam (live)":
        source = 0
    else:
        if uploaded_file is None:
            st.error("Please upload a video file first.")
            st.stop()
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            source = tfile.name

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"Error: Could not open video source '{source}'. Please check your webcam connection or file.")
        st.stop()

    st.toast("✅ Successfully connected to source")

    with st.spinner("Initializing Model..."):
        detector = Detector(conf_threshold=conf_threshold)
        tracker = CentroidTracker()
        st.toast("🎯 YOLO model loaded successfully!")

    id_to_class = {}
    id_to_conf = {}
    id_to_living = {}
    
    prev_time = time.time()
    start_time = time.time()
    frame_count = 0
    
    trajectory_history = defaultdict(list)
    event_log = []
    seen_objects = set()
    class_is_living = {}
    max_simultaneous = 0
    
    st.session_state.count_history = deque(maxlen=60)
    st.session_state.class_counts = defaultdict(int)
    st.session_state.living_total = 0
    st.session_state.nonliving_total = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.toast("🎬 End of video stream.")
                break
                
            frame_count += 1
            
            # Detect objects
            detections = detector.detect(frame)
            
            # Filter by allowed_classes
            filtered_detections = [d for d in detections if d["class_name"] in allowed_classes]

            # Helper to map bboxes back
            bbox_map = {tuple(d["bbox"]): (d["class_name"], d["confidence"], d["is_living"]) for d in filtered_detections}

            # Update tracker
            tracked_objects = tracker.update(filtered_detections)

            # Update metadata maps for drawing
            for obj_id, info in tracked_objects.items():
                cx, cy, bbox = info
                b = tuple(bbox)
                if b in bbox_map:
                    id_to_class[obj_id] = bbox_map[b][0]
                    id_to_conf[obj_id] = bbox_map[b][1]
                    id_to_living[obj_id] = bbox_map[b][2]
                
                if show_trails:
                    trajectory_history[obj_id].append((cx, cy))
                    if len(trajectory_history[obj_id]) > 30:
                        trajectory_history[obj_id].pop(0)

                if obj_id not in seen_objects:
                    seen_objects.add(obj_id)
                    cls_name = id_to_class.get(obj_id, "unknown")
                    
                    class_is_living[cls_name] = id_to_living.get(obj_id, False)
                    st.session_state.class_counts[cls_name] += 1
                    if class_is_living[cls_name]:
                        st.session_state.living_total += 1
                    else:
                        st.session_state.nonliving_total += 1
                        
                    event_log.append(f"Detected {cls_name} #{obj_id}")
                    if len(event_log) > 10:
                        event_log.pop(0)
                        
            # Calculate statistics
            living_counts = defaultdict(int)
            nonliving_counts = defaultdict(int)
            total_objs = len(tracked_objects)
            liv_count = 0
            nonliv_count = 0
            
            if total_objs > max_simultaneous:
                max_simultaneous = total_objs
                
            st.session_state.count_history.append((time.time(), total_objs))
            
            for obj_id in tracked_objects.keys():
                if obj_id in id_to_class:
                    cls_name = id_to_class[obj_id]
                    if id_to_living.get(obj_id, False):
                        living_counts[cls_name] += 1
                        liv_count += 1
                    else:
                        nonliving_counts[cls_name] += 1
                        nonliv_count += 1

            # Draw visualizations
            draw_detections(frame, tracked_objects, id_to_class, id_to_conf, id_to_living)
            
            if show_trails:
                for obj_id, pts in trajectory_history.items():
                    if obj_id in tracked_objects:
                        color = (0, 0, 220) if id_to_living.get(obj_id, False) else (160, 160, 160)
                        for i in range(1, len(pts)):
                            cv2.line(frame, pts[i-1], pts[i], color, 2)
            
            fps, prev_time = calculate_fps(prev_time)
            draw_fps(frame, fps)
            
            if show_stats:
                draw_stats(frame, living_counts, nonliving_counts)
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            video_placeholder.image(img, use_container_width=True)
            
            if frame_count == 1 or frame_count % 30 == 0:
                metric_total.metric("Total Objects", int(total_objs))
                metric_living.metric("Living Count", int(liv_count))
                metric_nonliving.metric("Non-living Count", int(nonliv_count))
                metric_fps.metric("FPS", int(fps))
                log_placeholder.text("\n".join(reversed(event_log)))
                
                try:
                    # Chart 1: Detection count over time
                    if len(st.session_state.count_history) > 0:
                        times, counts = zip(*st.session_state.count_history)
                        formatted_times = [time.strftime('%H:%M:%S', time.localtime(t)) for t in times]
                        fig1 = go.Figure(go.Scatter(
                            x=formatted_times, y=counts, mode='lines',
                            line=dict(color='#DC2626', width=2),
                            fill='tozeroy', fillcolor='rgba(220,38,38,0.1)'
                        ))
                        fig1.update_layout(
                            title="Objects detected (last 60s)",
                            showlegend=False,
                            margin=dict(l=0, r=0, t=30, b=0),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False)
                        )
                        chart1_placeholder.plotly_chart(fig1, use_container_width=True, key="line_chart")
                    
                    # Chart 2: Living vs Non-living ratio
                    fig2 = go.Figure(go.Pie(
                        labels=["Living", "Non-living"],
                        values=[int(st.session_state.living_total), int(st.session_state.nonliving_total)],
                        marker=dict(colors=["#DC2626", "#9CA3AF"]),
                        hole=0.55, textinfo='none'
                    ))
                    total_tracked = int(st.session_state.living_total + st.session_state.nonliving_total)
                    fig2.update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        annotations=[dict(text=str(total_tracked), x=0.5, y=0.5, font_size=20, showarrow=False)]
                    )
                    chart2_placeholder.plotly_chart(fig2, use_container_width=True, key="donut_chart")
                    
                    # Chart 3: Top 5 detected classes
                    if st.session_state.class_counts:
                        sorted_classes = sorted(st.session_state.class_counts.items(), key=lambda item: item[1], reverse=True)[:5]
                        sorted_classes.reverse()
                        class_names = [item[0].capitalize() for item in sorted_classes]
                        class_vals = [int(item[1]) for item in sorted_classes]
                        colors = ['#DC2626' if class_is_living.get(item[0], False) else '#9CA3AF' for item in sorted_classes]
                        
                        fig3 = go.Figure(go.Bar(
                            x=class_vals, y=class_names, orientation='h', marker_color=colors
                        ))
                        fig3.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis=dict(visible=False),
                            yaxis=dict(title="")
                        )
                        chart3_placeholder.plotly_chart(fig3, use_container_width=True, key="bar_chart")
                except Exception as e:
                    st.warning(f"Chart update failed: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    # 7. SESSION SUMMARY
    with st.expander("Session summary", expanded=True):
        runtime = int(time.time() - start_time)
        max_id = max(seen_objects) if seen_objects else 0
        most_detected = max(st.session_state.class_counts, key=st.session_state.class_counts.get) if st.session_state.class_counts else "None"
        
        st.write(f"**Total runtime:** {runtime} seconds")
        st.write(f"**Total unique objects tracked:** {int(max_id)}")
        st.write(f"**Most detected class:** {most_detected.capitalize() if most_detected != 'None' else 'None'}")
        st.write(f"**Peak simultaneous detection count:** {int(max_simultaneous)}")
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        try:
            if len(st.session_state.count_history) > 0:
                times, counts = zip(*st.session_state.count_history)
                formatted_times = [time.strftime('%H:%M:%S', time.localtime(t)) for t in times]
                fig1 = go.Figure(go.Scatter(x=formatted_times, y=counts, mode='lines', line=dict(color='#DC2626', width=2), fill='tozeroy', fillcolor='rgba(220,38,38,0.1)'))
                fig1.update_layout(title="Objects detected (last 60s)", showlegend=False, margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                col_sum1.plotly_chart(fig1, use_container_width=True, key="sum_line")
                
            fig2 = go.Figure(go.Pie(labels=["Living", "Non-living"], values=[int(st.session_state.living_total), int(st.session_state.nonliving_total)], marker=dict(colors=["#DC2626", "#9CA3AF"]), hole=0.55, textinfo='none'))
            total_tracked = int(st.session_state.living_total + st.session_state.nonliving_total)
            fig2.update_layout(title="Living vs Non-living ratio", showlegend=False, margin=dict(l=0, r=0, t=30, b=0), annotations=[dict(text=str(total_tracked), x=0.5, y=0.5, font_size=20, showarrow=False)])
            col_sum2.plotly_chart(fig2, use_container_width=True, key="sum_donut")
            
            if st.session_state.class_counts:
                sorted_classes = sorted(st.session_state.class_counts.items(), key=lambda item: item[1], reverse=True)[:5]
                sorted_classes.reverse()
                class_names = [item[0].capitalize() for item in sorted_classes]
                class_vals = [int(item[1]) for item in sorted_classes]
                colors = ['#DC2626' if class_is_living.get(item[0], False) else '#9CA3AF' for item in sorted_classes]
                fig3 = go.Figure(go.Bar(x=class_vals, y=class_names, orientation='h', marker_color=colors))
                fig3.update_layout(title="Top 5 detected classes", margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(visible=False), yaxis=dict(title=""))
                col_sum3.plotly_chart(fig3, use_container_width=True, key="sum_bar")
        except Exception as e:
            st.warning(f"Could not render summary charts: {e}")
