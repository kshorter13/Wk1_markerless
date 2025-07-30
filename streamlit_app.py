import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Try to fix MediaPipe model download path
def setup_mediapipe_env():
    """Set up environment for MediaPipe to use writable directories"""
    # Create a temporary directory for MediaPipe models
    model_dir = tempfile.mkdtemp(prefix="mediapipe_models_")
    
    # Set environment variables to redirect MediaPipe downloads
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    os.environ['GLOG_logtostderr'] = '1'
    os.environ['MEDIAPIPE_MODEL_PATH'] = model_dir
    
    return model_dir

# Set up MediaPipe environment before importing
model_dir = setup_mediapipe_env()

# Now try to import MediaPipe
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    st.error(f"MediaPipe import failed: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
    mp_drawing = None

class MediaPipeAnalyzer:
    def __init__(self):
        self.pose = None
        self.pose_initialized = False
        self.error_message = None
    
    def init_pose(self):
        """Initialize pose detection with error handling"""
        if not MEDIAPIPE_AVAILABLE:
            self.error_message = "MediaPipe not available"
            return False
            
        if not self.pose_initialized:
            try:
                # Try to create pose detector
                self.pose = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=0,  # Lightest model
                    enable_segmentation=False,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self.pose_initialized = True
                return True
            except Exception as e:
                self.error_message = f"Pose initialization failed: {str(e)}"
                # Try alternative approach with manual model handling
                try:
                    return self._try_alternative_init()
                except Exception as e2:
                    self.error_message = f"All initialization methods failed: {str(e2)}"
                    return False
        return True
    
    def _try_alternative_init(self):
        """Try alternative MediaPipe initialization"""
        try:
            # Try with different parameters
            self.pose = mp_pose.Pose(
                static_image_mode=False,  # Try dynamic mode
                model_complexity=1,       # Try medium complexity
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_initialized = True
            return True
        except:
            return False
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
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
        """Calculate knee angles only"""
        angles = {}
        
        try:
            # Left knee angle
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            if all(p.visibility > 0.3 for p in [left_hip, left_knee, left_ankle]):
                angles['left_knee'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Right knee angle
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            if all(p.visibility > 0.3 for p in [right_hip, right_knee, right_ankle]):
                angles['right_knee'] = self.calculate_angle(right_hip, right_knee, right_ankle)
                
        except Exception as e:
            pass
        
        return angles
    
    def process_frame(self, frame):
        """Process frame with MediaPipe"""
        if not self.init_pose():
            return frame, {}, False, self.error_message
        
        # Resize for performance
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Calculate knee angles
                angles = self.get_knee_angles(results.pose_landmarks.landmark)
                
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), angles, True, None
            else:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {}, False, "No pose detected"
                
        except Exception as e:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {}, False, f"Processing error: {str(e)}"

def get_video_info(video_path):
    """Get video information"""
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
        page_title="MediaPipe Knee Analysis",
        page_icon="🦵",
        layout="wide"
    )
    
    st.title("🦵 MediaPipe Knee Angle Analysis")
    st.markdown("*Advanced pose detection with knee angle measurements*")
    
    # Show MediaPipe status
    if MEDIAPIPE_AVAILABLE:
        st.success("✅ MediaPipe imported successfully")
    else:
        st.error("❌ MediaPipe import failed")
        st.stop()
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MediaPipeAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov'],
        help="Upload video for knee angle analysis"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            frame_count, fps = get_video_info(video_path)
            
            # Video info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Frames", frame_count)
            with col2:
                st.metric("FPS", f"{fps:.1f}")
            
            # Frame navigation
            current_frame = st.slider("Frame", 0, frame_count-1, 0)
            
            # Navigation controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("⏮️ Start"):
                    st.rerun()
            with col2:
                if st.button("⏪ -10"):
                    st.rerun()
            with col3:
                if st.button("⏩ +10"):
                    st.rerun()
            with col4:
                if st.button("⏭️ End"):
                    st.rerun()
            
            # Process frame
            frame = get_frame(video_path, current_frame)
            
            if frame is not None:
                # Initialize on first use
                if not st.session_state.analyzer.pose_initialized:
                    with st.spinner("🤖 Initializing MediaPipe pose detection..."):
                        init_success = st.session_state.analyzer.init_pose()
                        if not init_success:
                            st.error(f"❌ {st.session_state.analyzer.error_message}")
                            st.info("💡 MediaPipe may not work on Streamlit Cloud due to file permissions")
                            st.stop()
                        else:
                            st.success("✅ MediaPipe initialized successfully!")
                
                # Process frame
                with st.spinner("Processing frame..."):
                    processed_frame, angles, pose_detected, error = st.session_state.analyzer.process_frame(frame)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    current_time = current_frame / fps
                    st.image(
                        processed_frame,
                        caption=f"Frame {current_frame} | Time: {current_time:.2f}s",
                        use_container_width=True
                    )
                
                with col2:
                    st.subheader("🦵 Knee Angles")
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif pose_detected:
                        st.success("✅ Pose detected")
                        
                        # Display knee angles only
                        if 'left_knee' in angles:
                            st.metric("Left Knee", f"{angles['left_knee']:.1f}°")
                        else:
                            st.warning("Left knee: Not detected")
                            
                        if 'right_knee' in angles:
                            st.metric("Right Knee", f"{angles['right_knee']:.1f}°")
                        else:
                            st.warning("Right knee: Not detected")
                        
                        if not angles:
                            st.warning("No knee angles calculated")
                    else:
                        st.warning("⚠️ No pose detected in this frame")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            try:
                os.unlink(video_path)
            except:
                pass
    
    else:
        st.info("👆 Upload a video file to start MediaPipe joint analysis")
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown(f"""
            **MediaPipe Status:** {'✅ Available' if MEDIAPIPE_AVAILABLE else '❌ Not Available'}
            
            **Model Directory:** `{model_dir}`
            
            **Environment Variables Set:**
            - `MEDIAPIPE_DISABLE_GPU=1`
            - `GLOG_logtostderr=1` 
            - `MEDIAPIPE_MODEL_PATH={model_dir}`
            
            If MediaPipe still fails, it's likely due to Streamlit Cloud's file system restrictions.
            """)

if __name__ == "__main__":
    main()
