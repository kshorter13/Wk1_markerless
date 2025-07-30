import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Initialize MediaPipe only when needed
mp_pose = None
mp_drawing = None

def init_mediapipe():
    """Initialize MediaPipe only when needed"""
    global mp_pose, mp_drawing
    if mp_pose is None:
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            return True
        except Exception as e:
            st.error(f"MediaPipe initialization failed: {e}")
            return False
    return True

class LightMovementAnalyzer:
    def __init__(self):
        self.pose = None
        self.pose_initialized = False
    
    def init_pose(self):
        """Initialize pose detection when first needed"""
        if not self.pose_initialized:
            if init_mediapipe():
                try:
                    self.pose = mp_pose.Pose(
                        static_image_mode=True,
                        model_complexity=0,
                        enable_segmentation=False,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3
                    )
                    self.pose_initialized = True
                    return True
                except Exception as e:
                    st.error(f"Pose initialization failed: {e}")
                    return False
            return False
        return True
    
    def calculate_angle(self, p1, p2, p3):
        """Simple angle calculation"""
        try:
            a = np.array([p1.x, p1.y])
            b = np.array([p2.x, p2.y])
            c = np.array([p3.x, p3.y])
            
            ba = a - b
            bc = c - b
            
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine = np.clip(cosine, -1.0, 1.0)
            angle = np.arccos(cosine)
            return np.degrees(angle)
        except:
            return None
    
    def get_knee_angles(self, landmarks):
        """Get knee angles only"""
        left_angle = None
        right_angle = None
        
        try:
            # Left knee
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            if all(p.visibility > 0.3 for p in [left_hip, left_knee, left_ankle]):
                left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Right knee
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            if all(p.visibility > 0.3 for p in [right_hip, right_knee, right_ankle]):
                right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
                
        except:
            pass
        
        return left_angle, right_angle
    
    def process_frame_light(self, frame):
        """Lightweight frame processing"""
        # Initialize pose if needed
        if not self.init_pose():
            return frame, None, None, False
        
        # Resize for faster processing
        height, width = frame.shape[:2]
        if width > 480:
            scale = 480 / width
            new_width = 480
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.pose.process(rgb_frame)
        except Exception as e:
            st.error(f"Frame processing failed: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None, False
        
        left_angle = None
        right_angle = None
        pose_detected = False
        
        if results.pose_landmarks:
            pose_detected = True
            left_angle, right_angle = self.get_knee_angles(results.pose_landmarks.landmark)
            
            # Draw minimal landmarks
            try:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except:
                pass  # Skip drawing if it fails
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), left_angle, right_angle, pose_detected

def get_video_info(video_path):
    """Get basic video info"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps

def get_frame(video_path, frame_number):
    """Get single frame"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def main():
    st.set_page_config(
        page_title="Knee Angle Analyzer",
        page_icon="🦵",
        layout="centered"
    )
    
    st.title("🦵 Knee Angle Analyzer")
    st.markdown("*Lightweight movement analysis with delayed MediaPipe loading*")
    
    # Minimal session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LightMovementAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file for knee angle analysis"
    )
    
    if uploaded_file is not None:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Get video info
            frame_count, fps = get_video_info(video_path)
            
            # Basic info display
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Frames", frame_count)
            with col2:
                st.metric("Duration", f"{frame_count/fps:.1f}s")
            
            # Frame navigation
            st.subheader("Frame Navigation")
            current_frame = st.slider(
                "Frame",
                0, frame_count-1, 0,
                help="Navigate through video frames"
            )
            
            # Simple controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("⏮️ Start"):
                    current_frame = 0
                    st.rerun()
            with col2:
                if st.button("⏪ -10"):
                    current_frame = max(0, current_frame - 10)
                    st.rerun()
            with col3:
                if st.button("⏩ +10"):
                    current_frame = min(frame_count-1, current_frame + 10)
                    st.rerun()
            with col4:
                if st.button("⏭️ End"):
                    current_frame = frame_count-1
                    st.rerun()
            
            # Process current frame
            with st.spinner("Processing frame..."):
                frame = get_frame(video_path, current_frame)
                
                if frame is not None:
                    # Initialize MediaPipe on first use
                    if not st.session_state.analyzer.pose_initialized:
                        with st.spinner("Initializing MediaPipe (first time only)..."):
                            st.session_state.analyzer.init_pose()
                    
                    processed_frame, left_angle, right_angle, pose_detected = st.session_state.analyzer.process_frame_light(frame)
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        current_time = current_frame / fps
                        st.image(
                            processed_frame,
                            caption=f"Frame {current_frame} | Time: {current_time:.2f}s",
                            use_column_width=True
                        )
                    
                    with col2:
                        st.subheader("Knee Angles")
                        
                        # Show MediaPipe status
                        if st.session_state.analyzer.pose_initialized:
                            if pose_detected:
                                st.success("✅ Pose detected")
                                
                                # Left knee
                                if left_angle is not None:
                                    st.metric("Left Knee", f"{left_angle:.1f}°")
                                else:
                                    st.warning("Left knee: Not detected")
                                
                                # Right knee
                                if right_angle is not None:
                                    st.metric("Right Knee", f"{right_angle:.1f}°")
                                else:
                                    st.warning("Right knee: Not detected")
                                    
                            else:
                                st.warning("⚠️ No pose detected")
                                st.info("Try a frame with clearer body visibility")
                        else:
                            st.error("❌ MediaPipe initialization failed")
                            st.info("Check the error messages above")
                
                else:
                    st.error("Could not load frame")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        finally:
            # Always cleanup temp file
            try:
                os.unlink(video_path)
            except:
                pass
    
    else:
        # Simple instructions
        st.markdown("""
        ### 📋 How to Use:
        1. **Upload a video** (MP4, AVI, MOV)
        2. **Navigate frames** with the slider
        3. **View knee angles** in real-time
        
        ### 🎯 Features:
        - **Delayed MediaPipe loading** to avoid permission issues
        - **Lightweight processing** for Streamlit Cloud
        - **Knee angles only** (left & right)
        - **Error handling** for initialization problems
        
        ### 💡 Tips:
        - **Smaller videos** work best
        - **Clear body visibility** improves detection
        - **Side view** optimal for knee analysis
        - **First frame processing** may take longer (model download)
        """)

if __name__ == "__main__":
    main()
