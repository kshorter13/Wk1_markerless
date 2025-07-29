import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from typing import Tuple, Optional
import base64

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MovementAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """Calculate angle between three points (point2 is the vertex)"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range for arccos
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def get_knee_angle(self, landmarks, side='left') -> Optional[float]:
        """Calculate knee angle for specified side"""
        if side == 'left':
            hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
            knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
            ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
        else:
            hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
            knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE.value
            ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
        
        try:
            hip = np.array([landmarks[hip_idx].x, landmarks[hip_idx].y])
            knee = np.array([landmarks[knee_idx].x, landmarks[knee_idx].y])
            ankle = np.array([landmarks[ankle_idx].x, landmarks[ankle_idx].y])
            
            # Check if landmarks are visible
            if (landmarks[hip_idx].visibility < 0.5 or 
                landmarks[knee_idx].visibility < 0.5 or 
                landmarks[ankle_idx].visibility < 0.5):
                return None
            
            return self.calculate_angle(hip, knee, ankle)
        except:
            return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """Process a single frame and return annotated frame with knee angles"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate knee angles
            left_knee_angle = self.get_knee_angle(results.pose_landmarks.landmark, 'left')
            right_knee_angle = self.get_knee_angle(results.pose_landmarks.landmark, 'right')
            
            return annotated_frame, left_knee_angle, right_knee_angle
        
        return annotated_frame, None, None

def get_video_info(video_path: str) -> Tuple[int, float]:
    """Get video frame count and framerate"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps

def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def main():
    st.set_page_config(
        page_title="Movement Analysis App",
        page_icon="🏃‍♂️",
        layout="wide"
    )
    
    st.title("🏃‍♂️ Movement Analysis with MediaPipe")
    st.markdown("Upload a video to analyze movement patterns and calculate knee angles frame by frame.")
    
    # Initialize session state
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MovementAnalyzer()
    if 'frame_data' not in st.session_state:
        st.session_state.frame_data = {}
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze movement patterns"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            st.session_state.video_path = video_path
        
        # Get video information
        try:
            frame_count, fps = get_video_info(video_path)
            
            # Display video information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frame Count", frame_count)
            with col2:
                st.metric("Frame Rate", f"{fps:.2f} FPS")
            with col3:
                st.metric("Duration", f"{frame_count/fps:.2f} seconds")
            
            # Frame selection
            st.subheader("Frame Analysis")
            frame_number = st.slider(
                "Select Frame",
                min_value=0,
                max_value=frame_count-1,
                value=0,
                help="Use the slider to navigate through video frames"
            )
            
            # Play controls
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if st.button("⏮️ First"):
                    frame_number = 0
                    st.rerun()
            with col2:
                if st.button("⏪ -10"):
                    frame_number = max(0, frame_number - 10)
                    st.rerun()
            with col3:
                if st.button("⏪ -1"):
                    frame_number = max(0, frame_number - 1)
                    st.rerun()
            with col4:
                if st.button("⏩ +1"):
                    frame_number = min(frame_count-1, frame_number + 1)
                    st.rerun()
            with col5:
                if st.button("⏩ +10"):
                    frame_number = min(frame_count-1, frame_number + 10)
                    st.rerun()
            
            # Process current frame
            if frame_number not in st.session_state.frame_data:
                with st.spinner(f"Processing frame {frame_number}..."):
                    frame = extract_frame(video_path, frame_number)
                    if frame is not None:
                        annotated_frame, left_angle, right_angle = st.session_state.analyzer.process_frame(frame)
                        st.session_state.frame_data[frame_number] = {
                            'annotated_frame': annotated_frame,
                            'left_knee_angle': left_angle,
                            'right_knee_angle': right_angle
                        }
            
            # Display results
            if frame_number in st.session_state.frame_data:
                frame_data = st.session_state.frame_data[frame_number]
                
                # Display frame and angles
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"Frame {frame_number}")
                    st.image(
                        cv2.cvtColor(frame_data['annotated_frame'], cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_number} with pose landmarks",
                        use_column_width=True
                    )
                
                with col2:
                    st.subheader("Knee Angles")
                    
                    # Time information
                    current_time = frame_number / fps
                    st.info(f"⏰ Time: {current_time:.2f}s")
                    
                    # Left knee angle
                    if frame_data['left_knee_angle'] is not None:
                        st.metric(
                            "Left Knee Angle",
                            f"{frame_data['left_knee_angle']:.1f}°",
                            help="Angle between hip, knee, and ankle on the left side"
                        )
                    else:
                        st.warning("Left knee not detected")
                    
                    # Right knee angle
                    if frame_data['right_knee_angle'] is not None:
                        st.metric(
                            "Right Knee Angle",
                            f"{frame_data['right_knee_angle']:.1f}°",
                            help="Angle between hip, knee, and ankle on the right side"
                        )
                    else:
                        st.warning("Right knee not detected")
                    
                    # Angle interpretation
                    st.subheader("Interpretation")
                    if frame_data['left_knee_angle'] is not None or frame_data['right_knee_angle'] is not None:
                        st.markdown("""
                        **Knee Angle Guidelines:**
                        - 180° = Fully extended leg
                        - 90° = Right angle bend
                        - < 90° = Deep flexion
                        - Values closer to 180° indicate straighter leg
                        """)
            
            # Batch processing option
            st.subheader("Batch Analysis")
            if st.button("🔄 Process All Frames"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(0, frame_count, max(1, frame_count // 100)):  # Process every nth frame
                    if i not in st.session_state.frame_data:
                        frame = extract_frame(video_path, i)
                        if frame is not None:
                            annotated_frame, left_angle, right_angle = st.session_state.analyzer.process_frame(frame)
                            st.session_state.frame_data[i] = {
                                'annotated_frame': annotated_frame,
                                'left_knee_angle': left_angle,
                                'right_knee_angle': right_angle
                            }
                    
                    progress = i / frame_count
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {i}/{frame_count}")
                
                st.success("✅ Batch processing complete!")
            
            # Export data option
            if st.session_state.frame_data:
                st.subheader("Export Data")
                if st.button("📊 Export Angle Data as CSV"):
                    import pandas as pd
                    
                    data = []
                    for frame_num, frame_data in st.session_state.frame_data.items():
                        data.append({
                            'frame': frame_num,
                            'time_seconds': frame_num / fps,
                            'left_knee_angle': frame_data['left_knee_angle'],
                            'right_knee_angle': frame_data['right_knee_angle']
                        })
                    
                    df = pd.DataFrame(data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"knee_angles_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("Please make sure the uploaded file is a valid video format.")
    
    else:
        st.info("👆 Please upload a video file to get started")
        
        # Display sample instructions
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Upload a video**: Choose a video file (MP4, AVI, MOV, MKV)
            2. **Navigate frames**: Use the slider or control buttons to move through frames
            3. **View analysis**: See pose landmarks and knee angles for each frame
            4. **Batch process**: Process all frames for complete analysis
            5. **Export data**: Download knee angle measurements as CSV
            
            **Tips for best results:**
            - Use videos with clear view of the subject
            - Ensure good lighting and contrast
            - Subject should be clearly visible (not obscured)
            - Side view works best for knee angle analysis
            """)

if __name__ == "__main__":
    main()
