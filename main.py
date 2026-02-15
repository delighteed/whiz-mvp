import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np # type: ignore
import pygame # type: ignore
import time
from plyer import notification # type: ignore
import math

# Initialize pygame for sound
pygame.mixer.init()

# Загружаем звуки
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")
except:
    print("⚠️ alarm.wav file not found. Creating system beep")
    import array
    sample_rate = 22050
    frames = int(0.5 * sample_rate)
    arr = array.array('h', [0] * frames)
    for i in range(frames):
        arr[i] = int(32767.0 * np.sin(2.0 * np.pi * 440.0 * i / sample_rate))
    alarm_sound = pygame.mixer.Sound(buffer=arr)
    alarm_sound.set_volume(0.3)

# Создаем приятный звук для возвращения
try:
    return_sound = pygame.mixer.Sound("return.wav")
except:
    print("ℹ️ return.wav file not found. Creating pleasant return sound")
    import array
    sample_rate = 22050
    frames = int(0.4 * sample_rate)
    arr = array.array('h', [0] * frames)
    # Создаем приятный восходящий тон
    for i in range(frames):
        freq = 400 + (200 * i / frames)  # Восходящий тон от 400 до 600 Гц
        arr[i] = int(20000.0 * np.sin(2.0 * np.pi * freq * i / sample_rate))
    return_sound = pygame.mixer.Sound(buffer=arr)
    return_sound.set_volume(0.4)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==================== CONSTANTS AND SETTINGS ====================
# Working modes
MODES = {
    "relaxed": {"gaze_thresh": 0.25, "ear_thresh": 0.22, "sound": False, "head_rotation_time": 4.0},
    "standard": {"gaze_thresh": 0.20, "ear_thresh": 0.18, "sound": True, "head_rotation_time": 3.0},
    "focus": {"gaze_thresh": 0.15, "ear_thresh": 0.15, "sound": True, "head_rotation_time": 2.5},
    "gaming": {"gaze_thresh": 0.30, "ear_thresh": 0.25, "sound": False, "head_rotation_time": 5.0},
    "reading": {"gaze_thresh": 0.18, "ear_thresh": 0.16, "sound": False, "head_rotation_time": 3.5},
    "driving": {"gaze_thresh": 0.12, "ear_thresh": 0.12, "sound": True, "head_rotation_time": 1.5}
}

# Global settings
current_mode = "standard"
EAR_THRESHOLD = MODES[current_mode]["ear_thresh"]
GAZE_THRESHOLD = MODES[current_mode]["gaze_thresh"]
DISTRACT_BUFFER_TIME = 2.0
RETURN_TIME = 5.0
HAND_FACE_THRESHOLD = 0.15
BLINK_EAR_THRESHOLD = 0.21
EYES_CLOSED_LONG_TIME = 3.0

# Head rotation detection
HEAD_ROTATION_THRESHOLD = 45.0  # Increased to 45 degrees as requested

HEAD_ROTATION_DISTRACTION_TIME = MODES[current_mode]["head_rotation_time"]  # Время до отвлечения при повороте головы

# Interface states
show_face_mesh = True
show_pupils = True
show_hands = True
show_ui_panel = True
distracted = False
distracted_start_time = 0
buffer_start_time = 0
buffer_active = False
hands_near_face = False
hands_near_face_start = 0
hands_near_face_duration = 3.0
last_notification_time = 0
notification_cooldown = 10
blink_count = 0
blink_start_time = 0
eye_closed = False
eyes_closed_start_time = 0
fps_history = []

# Head rotation tracking
head_rotation_direction = "center"  # "center", "left", "right"
head_rotation_angle = 0.0  # угол поворота в градусах
head_rotation_start_time = 0
head_rotation_progress = 0
head_rotation_buffer_active = False

# Overload assessment
overload_level = "NORMAL"  # NORMAL, QUIET, CRITICAL
overload_score = 0
blink_rate = 0
distraction_count = 0
last_blink_time = 0
blink_times = []
distraction_times = []
overload_update_time = 0

# Screen notification system
screen_notifications = []
notification_duration = 8.0
enable_system_notifications = True
last_system_notification_time = 0

# Sound flags
played_return_sound = False

# Face landmarks indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# ==================== AI LOAD CLASSIFIER ====================
from sklearn.ensemble import RandomForestClassifier
from collections import deque

class LoadClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_trained = False
        self.training_data = [] # List of (features, label)
        self.feature_buffer = deque(maxlen=30) # Store raw metrics
        self.sample_cooldown = 0
        
    def add_sample(self, raw_metrics, label):
        """
        Add sample for calibration.
        Uses a sliding window approach with stride 5 to get more data.
        """
        self.feature_buffer.append(raw_metrics)
        self.sample_cooldown += 1
        
        # Only add sample every 5 frames to avoid excessive correlation but ensure enough data
        if len(self.feature_buffer) == 30 and self.sample_cooldown >= 5:
            features = self._extract_features()
            self.training_data.append((features, label))
            self.sample_cooldown = 0
            return True # Sample added
        return False
        
    def train(self):
        if len(self.training_data) < 10: # Minimum samples total
            return False, f"Not enough data ({len(self.training_data)} samples)"
            
        X = [d[0] for d in self.training_data]
        y = [d[1] for d in self.training_data]
        
        # Check class balance
        unique_classes = set(y)
        if len(unique_classes) < 2:
            return False, f"Need both Relaxed and Focused examples (got {unique_classes})"
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True, f"Trained on {len(X)} samples"
        except Exception as e:
            return False, str(e)
            
    def predict(self, raw_metrics):
        self.feature_buffer.append(raw_metrics)
        if len(self.feature_buffer) < 30:
            return 0.5 # Not enough history
            
        features = self._extract_features()
        if not self.is_trained:
            return 0.5 # Default unsure
            
        try:
            probs = self.model.predict_proba([features])[0]
            # probs[1] is probability of class 1 (Focused/Loaded)
            return probs[1] 
        except:
            return 0.5

    def _extract_features(self):
        # Convert buffer to numpy array for stats
        ears = [m['ear'] for m in self.feature_buffer]
        gaze_xs = [m['gaze_x'] for m in self.feature_buffer]
        hand_moving = [m['hand_active'] for m in self.feature_buffer]
        
        # Extended feature set
        return [
            np.mean(ears),
            np.std(ears),
            np.min(ears),
            np.max(ears),
            np.mean(gaze_xs),
            np.std(gaze_xs),
            np.mean(hand_moving) # % of time hands are active
        ]

# ==================== FUNCTIONS ====================
def calculate_ear(eye_points, landmarks):
    """Calculate Eye Aspect Ratio"""
    try:
        A = np.linalg.norm(np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y]) - 
                           np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y]))
        B = np.linalg.norm(np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y]) - 
                           np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y]))
        C = np.linalg.norm(np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y]) - 
                           np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y]))
        return (A + B) / (2.0 * C + 1e-6)
    except:
        return 0.25

def calculate_gaze_direction(landmarks):
    """Calculate gaze direction"""
    try:
        left_eye_center = (landmarks[33].x + landmarks[133].x) / 2
        right_eye_center = (landmarks[362].x + landmarks[263].x) / 2
        gaze_x = (left_eye_center + right_eye_center) / 2
        return gaze_x
    except:
        return 0.5

def detect_head_rotation(landmarks):
    """Detect head rotation (yaw) based on 3D face landmarks"""
    try:
        # Используем ключевые точки лица для определения поворота
        # Левая и правая сторона лица
        left_cheek = landmarks[234]  # Левая щека
        right_cheek = landmarks[454]  # Правая щека
        nose_tip = landmarks[1]  # Кончик носа
        
        # Координаты точек
        lx, ly = left_cheek.x, left_cheek.y
        rx, ry = right_cheek.x, right_cheek.y
        nx, ny = nose_tip.x, nose_tip.y
        
        # Векторы от носа к щекам
        vec_left = np.array([lx - nx, ly - ny])
        vec_right = np.array([rx - nx, ry - ny])
        
        # Длины векторов
        len_left = np.linalg.norm(vec_left)
        len_right = np.linalg.norm(vec_right)
        
        if len_left > 0 and len_right > 0:
            # Нормализуем векторы
            vec_left_norm = vec_left / len_left
            vec_right_norm = vec_right / len_right
            
            # Определяем разницу в длинах (при повороте одна сторона становится короче)
            length_ratio = len_left / len_right
            
            # Угол поворота (в градусах)
            # При length_ratio > 1 - поворот вправо, < 1 - поворот влево
            if length_ratio >= 1.0:
                angle = (length_ratio - 1.0) * 45.0
            else:
                angle = (1.0 - (1.0 / length_ratio)) * 45.0
        
        # Определяем направление
        if angle < -HEAD_ROTATION_THRESHOLD:
            return "left", abs(angle)
        elif angle > HEAD_ROTATION_THRESHOLD:
            return "right", abs(angle)
        else:
            return "center", 0
    except:
        return "center", 0

def track_pupils(landmarks):
    """Track pupil positions"""
    pupils = {"left": {"x": 0.5, "y": 0.5, "detected": False},
              "right": {"x": 0.5, "y": 0.5, "detected": False}}
    
    try:
        if len(landmarks) >= 478:
            # Left pupil
            left_points = []
            for idx in LEFT_IRIS_INDICES:
                if idx < len(landmarks):
                    left_points.append([landmarks[idx].x, landmarks[idx].y])
            if left_points:
                left_points = np.array(left_points)
                pupils["left"]["x"] = float(np.mean(left_points[:, 0]))
                pupils["left"]["y"] = float(np.mean(left_points[:, 1]))
                pupils["left"]["detected"] = True
            
            # Right pupil
            right_points = []
            for idx in RIGHT_IRIS_INDICES:
                if idx < len(landmarks):
                    right_points.append([landmarks[idx].x, landmarks[idx].y])
            if right_points:
                right_points = np.array(right_points)
                pupils["right"]["x"] = float(np.mean(right_points[:, 0]))
                pupils["right"]["y"] = float(np.mean(right_points[:, 1]))
                pupils["right"]["detected"] = True
    except:
        pass
    
    return pupils

def check_hands_near_face(hand_landmarks, face_landmarks):
    """Check if hands are near face"""
    if not hand_landmarks or not face_landmarks:
        return False
    
    try:
        face_center_x = face_landmarks[1].x
        face_center_y = face_landmarks[1].y
        
        for hand in hand_landmarks:
            wrist = hand.landmark[0]
            dist = np.sqrt((wrist.x - face_center_x)**2 + (wrist.y - face_center_y)**2)
            if dist < HAND_FACE_THRESHOLD:
                return True
    except:
        pass
    
    return False

def detect_blink(ear_value):
    """Detect blink and count"""
    global blink_count, eye_closed, blink_start_time, last_blink_time, blink_times, eyes_closed_start_time
    
    current_time = time.time()
    
    if ear_value < BLINK_EAR_THRESHOLD and not eye_closed:
        eye_closed = True
        blink_start_time = current_time
        if eyes_closed_start_time == 0:
            eyes_closed_start_time = current_time
    elif ear_value >= BLINK_EAR_THRESHOLD and eye_closed:
        eye_closed = False
        if current_time - blink_start_time < 0.5:
            blink_count += 1
            last_blink_time = current_time
            blink_times.append(current_time)
        blink_start_time = 0
        eyes_closed_start_time = 0
    
    return eye_closed

def update_overload_assessment():
    """Update overload assessment based on recent activity"""
    global overload_level, overload_score, blink_rate, distraction_count, blink_times, distraction_times
    
    current_time = time.time()
    
    # Keep only events from last 60 seconds
    blink_times = [t for t in blink_times if current_time - t < 60]
    distraction_times = [t for t in distraction_times if current_time - t < 60]
    
    # Calculate blink rate (blinks per minute)
    if len(blink_times) > 1:
        time_span = blink_times[-1] - blink_times[0]
        if time_span > 0:
            blink_rate = len(blink_times) / (time_span / 60)
        else:
            blink_rate = 0
    else:
        blink_rate = 0
    
    # Calculate distraction frequency
    distraction_count = len(distraction_times)
    
    # Calculate overload score (0-100)
    score = 0
    
    # Blink rate component (40% weight)
    if blink_rate > 30:
        score += 40
    elif blink_rate > 20:
        score += 20
    elif blink_rate < 10:
        score += 10
    
    # Distraction frequency component (40% weight)
    score += min(40, distraction_count * 10)
    
    # Current status component (20% weight)
    if distracted:
        score += 20
    
    overload_score = min(100, score)
    
    # Determine level
    if overload_score > 70:
        overload_level = "CRITICAL"
    elif overload_score > 40:
        overload_level = "NORMAL"
    else:
        overload_level = "QUIET"
    
    return overload_level, overload_score

def add_screen_notification(message, level="info"):
    """Add a notification to be displayed on screen"""
    global screen_notifications
    
    # Remove old notifications
    current_time = time.time()
    screen_notifications = [n for n in screen_notifications 
                          if current_time - n["time"] < notification_duration]
    
    # Add new notification
    screen_notifications.append({
        "text": message,
        "time": current_time,
        "level": level
    })

def draw_progress_bar(frame, progress, x=50, y=400, width=300, height=30):
    """Draw progress bar with fixed maximum"""
    # Ограничиваем прогресс 1.0
    progress = min(1.0, progress)
    
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    fill_width = int(width * progress)
    
    if progress < 0.3:
        color = (0, 0, 255)  # Red
    elif progress < 0.7:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 255, 0)  # Green
    
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
    
    if progress < 1.0:
        time_left = max(0, RETURN_TIME * (1 - progress))
        text = f"Return in: {time_left:.1f}s"
    else:
        text = "RETURNED!"
    
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_ui_panel(frame, metrics, h, w, ai_level=0.5, calibration_active=False, calibration_label=None):
    """Draw control panel and info"""
    if not show_ui_panel:
        return frame
    
    # Panel background
    panel_height = 310  # Increased for head rotation info
    cv2.rectangle(frame, (10, 10), (480, panel_height), (20, 20, 30), -1)
    cv2.rectangle(frame, (10, 10), (480, panel_height), (100, 100, 100), 2)
    
    # Header
    mode_color = (0, 255, 100)
    cv2.putText(frame, f"MODE: {current_mode.upper()}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    # Status
    status = "DISTRACTED" if distracted else "FOCUSED"
    status_color = (0, 100, 255) if distracted else (100, 255, 100)
    cv2.putText(frame, f"STATUS: {status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Overload assessment
    overload_color = (0, 255, 0)
    if overload_level == "NORMAL":
        overload_color = (0, 165, 255)
    elif overload_level == "CRITICAL":
        overload_color = (0, 0, 255)
    
    cv2.putText(frame, f"OVERLOAD: {overload_level}", (20, 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, overload_color, 2)
    cv2.putText(frame, f"Score: {overload_score}/100", (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Head rotation info
    head_color = (100, 255, 100) if head_rotation_direction == "center" else (0, 165, 255)
    direction_text = f"HEAD ROTATION: {head_rotation_direction.upper()}"
    if head_rotation_direction != "center":
        direction_text += f" ({head_rotation_angle:.1f}°)"
    
    cv2.putText(frame, direction_text, (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2)
    
    # Metrics
    y_pos = 190
    metric_lines = [
        f"EAR: {metrics.get('ear', 0.25):.3f}",
        f"Gaze X: {metrics.get('gaze_x', 0.5):.3f}",
        f"Blinks: {blink_count} ({blink_rate:.1f}/min)",
        f"AI Load: {ai_level*100:.0f}% {'(CALIBRATING)' if calibration_active else ''}",
        f"Hands near face: {'YES' if hands_near_face else 'NO'}",
        f"Distractions: {distraction_count}/min"
    ]
    
    for i, line in enumerate(metric_lines):
        cv2.putText(frame, line, (20, y_pos + i * 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    # Interface settings
    settings_y = y_pos + len(metric_lines) * 25 + 10
    settings = [
        f"Face mesh: {'ON' if show_face_mesh else 'OFF'}",
        f"Pupils: {'ON' if show_pupils else 'OFF'}",
        f"Hands: {'ON' if show_hands else 'OFF'}",
        f"Sound: {'ON' if MODES[current_mode]['sound'] else 'OFF'}",
        f"Sys notif: {'ON' if enable_system_notifications else 'OFF'}"
    ]
    
    for i, setting in enumerate(settings):
        cv2.putText(frame, setting, (20, settings_y + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
    
    # Distraction buffer
    if buffer_active and not distracted:
        buffer_progress = (time.time() - buffer_start_time) / DISTRACT_BUFFER_TIME
        bar_x, bar_y = 20, settings_y + 100
        bar_width = 200
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (50, 50, 50), -1)
        fill_width = int(bar_width * buffer_progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 15), (0, 165, 255), -1)
        cv2.putText(frame, f"Buffer: {buffer_progress*100:.0f}%", (bar_x + bar_width + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Head rotation buffer
    if head_rotation_buffer_active and not distracted:
        head_progress = (time.time() - head_rotation_start_time) / HEAD_ROTATION_DISTRACTION_TIME
        head_progress = min(1.0, head_progress)
        bar_x, bar_y = 20, settings_y + 120
        bar_width = 200
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (50, 50, 50), -1)
        fill_width = int(bar_width * head_progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 15), (255, 165, 0), -1)
        cv2.putText(frame, f"Head rotation: {head_progress*100:.0f}%", (bar_x + bar_width + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Controls
    controls_y = h - 120
    cv2.rectangle(frame, (10, controls_y), (w - 10, h - 10), (30, 30, 40), -1)
    
    controls = [
        "CONTROLS:",
        "1-6: Modes (1-relaxed, 6-driving)",
        "F: Face mesh  H: Hands  P: Pupils",
        "U: UI panel  S: Sound  N: Sys notif",
        "R: Reset  Q: Quit  Z: Clear blinks",
        "C: Calibrate Relaxed  V: Calibrate Focused"
    ]
    
    for i, control in enumerate(controls):
        y = controls_y + 25 + i * 20
        color = (255, 255, 200) if i == 0 else (150, 150, 150)
        cv2.putText(frame, control, (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def draw_screen_notifications(frame, h, w):
    """Draw notifications on screen (top-right corner)"""
    current_time = time.time()
    
    # Filter out old notifications
    global screen_notifications
    screen_notifications = [n for n in screen_notifications 
                          if current_time - n["time"] < notification_duration]
    
    if not screen_notifications:
        return
    
    # Draw notifications from top to bottom
    start_x = w - 450
    start_y = 50
    spacing = 35
    
    # Limit number of notifications to 5
    visible_notifications = screen_notifications[-5:]
    
    for i, notif in enumerate(visible_notifications):
        y_pos = start_y + i * spacing
        
        # Calculate fade effect
        time_passed = current_time - notif["time"]
        fade = 1.0
        if notification_duration - time_passed < 2.0:
            fade = (notification_duration - time_passed) / 2.0
        
        # Set color based on level
        if notif["level"] == "alert":
            bg_color = (0, 0, int(200 * fade))
            text_color = (int(255 * fade), int(100 * fade), int(100 * fade))
            border_color = (int(255 * fade), 0, 0)
        elif notif["level"] == "warning":
            bg_color = (0, int(100 * fade), int(200 * fade))
            text_color = (int(255 * fade), int(200 * fade), int(100 * fade))
            border_color = (int(255 * fade), int(165 * fade), 0)
        else:
            bg_color = (int(40 * fade), int(40 * fade), int(60 * fade))
            text_color = (int(200 * fade), int(200 * fade), int(255 * fade))
            border_color = (int(100 * fade), int(100 * fade), int(200 * fade))
        
        # Get text size
        text_size = cv2.getTextSize(notif["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_width = min(text_size[0] + 20, 400)
        
        # Draw background
        cv2.rectangle(frame, 
                     (start_x - 10, y_pos - 25), 
                     (start_x + text_width, y_pos + 5), 
                     bg_color, -1)
        
        # Draw border
        cv2.rectangle(frame, 
                     (start_x - 10, y_pos - 25), 
                     (start_x + text_width, y_pos + 5), 
                     border_color, 2)
        
        # Draw text (без смайликов)
        cv2.putText(frame, notif["text"], (start_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

def send_notification():
    """Send system notification"""
    global last_notification_time, last_system_notification_time, distraction_times
    
    current_time = time.time()
    
    # Add to distraction history
    distraction_times.append(current_time)
    
    # Screen notification
    if current_time - last_notification_time > notification_cooldown:
        last_notification_time = current_time
        add_screen_notification(f"ALERT: Distracted in {current_mode} mode!", "alert")
    
    # System notification
    if enable_system_notifications and current_time - last_system_notification_time > 30:
        try:
            notification.notify(
                title='EYE TRACKER ALERT',
                message=f'Distracted in {current_mode.upper()} mode! Focus required!',
                app_name='Eye Tracker Pro',
                timeout=5
            )
            last_system_notification_time = current_time
            add_screen_notification("System notification sent", "info")
        except Exception as e:
            print(f"System notification error: {e}")
            add_screen_notification("System notifications disabled", "warning")

def play_sound(sound_type="alert"):
    """Play sound alert or return sound"""
    if sound_type == "alert" and alarm_sound and MODES[current_mode]["sound"]:
        try:
            alarm_sound.play()
        except:
            print("\a", end="", flush=True)
    elif sound_type == "return" and return_sound and MODES[current_mode]["sound"]:
        try:
            return_sound.play()
        except:
            print("\a", end="", flush=True)

def switch_mode(new_mode):
    """Switch working mode"""
    global current_mode, EAR_THRESHOLD, GAZE_THRESHOLD, HEAD_ROTATION_DISTRACTION_TIME
    
    if new_mode in MODES:
        current_mode = new_mode
        EAR_THRESHOLD = MODES[new_mode]["ear_thresh"]
        GAZE_THRESHOLD = MODES[new_mode]["gaze_thresh"]
        HEAD_ROTATION_DISTRACTION_TIME = MODES[new_mode]["head_rotation_time"]
        
        print(f"\nMode changed: {new_mode.upper()}")
        print(f"   EAR threshold: {EAR_THRESHOLD}")
        print(f"   Gaze threshold: {GAZE_THRESHOLD}")
        print(f"   Head rotation time: {HEAD_ROTATION_DISTRACTION_TIME}s")
        print(f"   Sound: {'ON' if MODES[new_mode]['sound'] else 'OFF'}")
        
        add_screen_notification(f"Mode changed to {new_mode.upper()}", "info")

# ==================== MAIN FUNCTION ====================
def main():
    global show_face_mesh, show_pupils, show_hands, show_ui_panel
    global distracted, distracted_start_time, buffer_start_time, buffer_active
    global hands_near_face, hands_near_face_start, current_mode, blink_count
    global overload_update_time, enable_system_notifications, eyes_closed_start_time
    global head_rotation_direction, head_rotation_angle, head_rotation_start_time, head_rotation_buffer_active
    global played_return_sound
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("=" * 70)
    print("SMART EYE TRACKER - Professional Edition")
    print("=" * 70)
    print("\nCONTROLS:")
    print("1-6: Modes (Relaxed, Standard, Focus, Gaming, Reading, Driving)")
    print("F: Face mesh  H: Hands  P: Pupils")
    print("U: UI panel  S: Sound toggle  N: System notifications")
    print("R: Reset state  Z: Clear blink counter  Q: Quit")
    print("C: Calibrate Relaxed  V: Calibrate Focused")
    print("=" * 70)
    print(f"\nStarting in mode: {current_mode.upper()}")
    
    # Initial notification
    add_screen_notification("Eye Tracker started!", "info")
    add_screen_notification(f"Mode: {current_mode.upper()}", "info")
    add_screen_notification("Press U to toggle UI panel", "info")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as face_mesh, \
        mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        
        # AI Classifier Initialization
        load_classifier = LoadClassifier()
        calibration_active = False
        calibration_label = None # 0: Relaxed, 1: Focused
        calibration_start = 0
        calibration_samples = 0
        ai_load_level = 0.5
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face and hands
            face_results = face_mesh.process(frame_rgb)
            hand_results = hands.process(frame_rgb)
            
            current_time = time.time()
            gaze_distracted = False
            eyes_closed_long = False
            head_rotation_distracted = False
            metrics = {"ear": 0.25, "gaze_x": 0.5, "pupils_detected": False}
            pupils_data = {"left": {"x": 0.5, "y": 0.5}, "right": {"x": 0.5, "y": 0.5}}
            
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Calculate metrics
                    left_ear = calculate_ear(LEFT_EYE_INDICES, landmarks)
                    right_ear = calculate_ear(RIGHT_EYE_INDICES, landmarks)
                    avg_ear = (left_ear + right_ear) / 2
                    metrics["ear"] = avg_ear
                    
                    gaze_x = calculate_gaze_direction(landmarks)
                    metrics["gaze_x"] = gaze_x
                    
                    # Detect head rotation (улучшенная версия - ПОВОРОТ головы)
                    head_direction, head_angle = detect_head_rotation(landmarks)
                    head_rotation_direction = head_direction
                    head_rotation_angle = head_angle
                    
                    # Track pupils
                    pupils = track_pupils(landmarks)
                    pupils_data = pupils
                    metrics["pupils_detected"] = pupils["left"]["detected"] or pupils["right"]["detected"]
                    
                    # Detect blink
                    detect_blink(avg_ear)
                    
                    # Check gaze distraction
                    # Modified: Require BOTH pupils to be out of bounds if detected
                    if pupils["left"]["detected"] and pupils["right"]["detected"]:
                        left_out = abs(pupils["left"]["x"] - 0.5) > GAZE_THRESHOLD
                        right_out = abs(pupils["right"]["x"] - 0.5) > GAZE_THRESHOLD
                        gaze_distracted = left_out and right_out
                    else:
                        gaze_distracted = abs(gaze_x - 0.5) > GAZE_THRESHOLD
                    
                    # Check if eyes are closed for distraction
                    eyes_closed_now = avg_ear < EAR_THRESHOLD
                    
                    if eyes_closed_now and eyes_closed_start_time == 0:
                        eyes_closed_start_time = current_time
                    elif not eyes_closed_now:
                        eyes_closed_start_time = 0
                    
                    if eyes_closed_start_time != 0 and (current_time - eyes_closed_start_time >= EYES_CLOSED_LONG_TIME):
                        eyes_closed_long = True
                    
                    # Check hands near face
                    if hand_results.multi_hand_landmarks:
                        hands_near_face = check_hands_near_face(
                            hand_results.multi_hand_landmarks, 
                            landmarks
                        )
                        if hands_near_face:
                            if hands_near_face_start == 0:
                                hands_near_face_start = current_time
                                add_screen_notification("Hands detected near face", "warning")
                        else:
                            hands_near_face_start = 0
                    
                    # Draw face mesh
                    if show_face_mesh:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=1
                            )
                        )
                    
                    # Draw pupils
                    if show_pupils and pupils["left"]["detected"]:
                        lx = int(pupils["left"]["x"] * w)
                        ly = int(pupils["left"]["y"] * h)
                        cv2.circle(frame, (lx, ly), 8, (0, 255, 255), -1)
                        cv2.circle(frame, (lx, ly), 10, (0, 255, 255), 2)
                    
                    if show_pupils and pupils["right"]["detected"]:
                        rx = int(pupils["right"]["x"] * w)
                        ry = int(pupils["right"]["y"] * h)
                        cv2.circle(frame, (rx, ry), 8, (255, 255, 0), -1)
                        cv2.circle(frame, (rx, ry), 10, (255, 255, 0), 2)
            
            # Draw hands
            if show_hands and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
            
            # ================== AI LOGIC ==================
            # Prepare metrics for AI
            hand_active_val = 1.0 if hand_results.multi_hand_landmarks else 0.0
            raw_metrics = {
                'ear': metrics["ear"],
                'gaze_x': metrics["gaze_x"],
                'hand_active': hand_active_val
            }
            
            if calibration_active:
                if time.time() - calibration_start < 10:
                    added = load_classifier.add_sample(raw_metrics, calibration_label)
                    if added:
                        calibration_samples += 1
                else:
                    calibration_active = False
                    add_screen_notification(f"Calibration done ({calibration_samples} samples)", "info")
                    # Auto-train attempt
                    labels = [d[1] for d in load_classifier.training_data]
                    if 0 in labels and 1 in labels:
                        success, msg = load_classifier.train()
                        add_screen_notification(f"AI Training: {msg}", "info")
            
            if not calibration_active and load_classifier.is_trained:
                ai_load_level = load_classifier.predict(raw_metrics)
            # ==============================================
            
            # Update overload assessment every 2 seconds
            
            # Update overload assessment every 2 seconds
            if current_time - overload_update_time > 2.0:
                update_overload_assessment()
                overload_update_time = current_time
            
            # Head rotation distraction logic - ПОВОРОТ ГОЛОВЫ
            if head_rotation_direction != "center" and not hands_near_face:
                if head_rotation_start_time == 0:
                    head_rotation_start_time = current_time
                    head_rotation_buffer_active = True
                    add_screen_notification(f"Head rotated {head_rotation_direction}! ({head_rotation_angle:.1f}°)", "warning")
                
                head_rotation_elapsed = current_time - head_rotation_start_time
                if head_rotation_elapsed >= HEAD_ROTATION_DISTRACTION_TIME:
                    head_rotation_distracted = True
            else:
                if head_rotation_start_time != 0:
                    head_rotation_start_time = 0
                    head_rotation_buffer_active = False
                    if not distracted:
                        add_screen_notification("Head centered", "info")
            
            # Distraction logic - now includes head rotation and user presence
            user_present = bool(face_results.multi_face_landmarks)
            
            if not user_present:
                is_distracted_now = True
                # Optional: Add notification for absence? 
                # add_screen_notification("User away!", "warning")
            else:
                is_distracted_now = (eyes_closed_long or gaze_distracted or head_rotation_distracted) and not hands_near_face
            
            if is_distracted_now and not buffer_active and not distracted:
                buffer_start_time = current_time
                buffer_active = True
                if eyes_closed_long:
                    add_screen_notification(f"Eyes closed for {EYES_CLOSED_LONG_TIME}s! Buffer started...", "warning")
                elif gaze_distracted:
                    add_screen_notification("Gaze distracted! Buffer started...", "warning")
                elif head_rotation_distracted:
                    add_screen_notification(f"Head rotated {head_rotation_direction} for {HEAD_ROTATION_DISTRACTION_TIME}s! Buffer started...", "warning")
            
            elif not is_distracted_now and buffer_active:
                buffer_active = False
                add_screen_notification("Distraction buffer cleared", "info")
            
            if buffer_active and not distracted:
                buffer_progress = (current_time - buffer_start_time) / DISTRACT_BUFFER_TIME
                if buffer_progress >= 1.0:
                    distracted = True
                    distracted_start_time = current_time
                    buffer_active = False
                    played_return_sound = False  # Сброс флага звука
                    
                    if eyes_closed_long:
                        print(f"ALERT! Eyes closed for {EYES_CLOSED_LONG_TIME+DISTRACT_BUFFER_TIME}s!")
                    elif gaze_distracted:
                        print("ALERT! Gaze distracted!")
                    elif head_rotation_distracted:
                        print(f"ALERT! Head rotated {head_rotation_direction} for {HEAD_ROTATION_DISTRACTION_TIME}s!")
                    
                    play_sound("alert")
                    send_notification()
            
            # Modified Return Logic
            if distracted:
                if not is_distracted_now:
                    # User is focused, start counting
                    if focus_start_time == 0:
                        focus_start_time = current_time
                    
                    # Calculate progress based on how long we've been FOCUSED
                    # Using a fixed 2.0s return time instead of RETURN_TIME which might be long
                    REQUIRED_FOCUS_TIME = 2.0 
                    return_progress = (current_time - focus_start_time) / REQUIRED_FOCUS_TIME
                    return_progress = min(1.0, return_progress)
                    
                    if return_progress >= 1.0:
                        distracted = False
                        distracted_start_time = 0
                        focus_start_time = 0
                        print("Returned to focus!")
                        add_screen_notification("Returned to focus!", "info")
                        
                        if not played_return_sound:
                            play_sound("return")
                            played_return_sound = True
                else:
                    # User is distracted, reset focus timer
                    focus_start_time = 0
            
            # Draw UI panel
            frame = draw_ui_panel(frame, metrics, h, w, ai_level=ai_load_level, 
                                 calibration_active=calibration_active, 
                                 calibration_label=calibration_label)
            
            # Draw screen notifications (без смайликов)
            draw_screen_notifications(frame, h, w)
            
            # Distraction progress bar
            if distracted and focus_start_time > 0:
                # Show return progress only when focused
                duration = current_time - focus_start_time
                return_progress = duration / 2.0 # 2.0s required
                draw_progress_bar(frame, return_progress, 50, h//2, w-100, 30)
            elif distracted:
                 # Show full red bar or text?
                 cv2.putText(frame, "DISTRACTED", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Blink indicator
            if eye_closed:
                cv2.putText(frame, "BLINK!", (w//2 - 50, h//2 - 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            # Overload warning
            if overload_level == "CRITICAL":
                cv2.putText(frame, "CRITICAL OVERLOAD!", (w//2 - 150, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Eyes closed timer indicator
            if eyes_closed_start_time != 0 and not distracted:
                closed_duration = current_time - eyes_closed_start_time
                if closed_duration > 1.0:
                    timer_text = f"Eyes closed: {closed_duration:.1f}s/{EYES_CLOSED_LONG_TIME}s"
                    cv2.putText(frame, timer_text, (w//2 - 100, h//2 + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Head rotation indicator
            if head_rotation_direction != "center" and not distracted:
                if head_rotation_start_time != 0:
                    rotation_duration = current_time - head_rotation_start_time
                    time_left = max(0, HEAD_ROTATION_DISTRACTION_TIME - rotation_duration)
                    
                    # Рисуем индикатор в зависимости от стороны
                    if head_rotation_direction == "left":
                        indicator_text = f"HEAD ROTATED LEFT: {time_left:.1f}s ({head_rotation_angle:.1f}°)"
                        color = (0, 165, 255)  # Оранжевый для лево
                        # Стрелка влево
                        cv2.putText(frame, "<--", (50, h//2 + 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    else:  # right
                        indicator_text = f"HEAD ROTATED RIGHT: {time_left:.1f}s ({head_rotation_angle:.1f}°)"
                        color = (255, 165, 0)  # Желто-оранжевый для право
                        # Стрелка вправо
                        cv2.putText(frame, "-->", (w - 150, h//2 + 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
                    # Текст индикатора
                    cv2.putText(frame, indicator_text, (w//2 - 200, h//2 + 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # FPS
            frame_count += 1
            if current_time - last_fps_update >= 1.0:
                fps = frame_count / (current_time - last_fps_update)
                fps_history.append(fps)
                if len(fps_history) > 10:
                    fps_history.pop(0)
                frame_count = 0
                last_fps_update = current_time
            
            avg_fps = np.mean(fps_history) if fps_history else 0
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 120, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow(f'Eye Tracker Pro - {current_mode.upper()}', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif ord('1') <= key <= ord('6'):
                mode_index = key - ord('1')
                modes = ["relaxed", "standard", "focus", "gaming", "reading", "driving"]
                if mode_index < len(modes):
                    switch_mode(modes[mode_index])
            
            elif key == ord('f'):
                show_face_mesh = not show_face_mesh
                status = "ON" if show_face_mesh else "OFF"
                print(f"Face mesh: {status}")
                add_screen_notification(f"Face mesh: {status}", "info")
            
            elif key == ord('p'):
                show_pupils = not show_pupils
                status = "ON" if show_pupils else "OFF"
                print(f"Pupils: {status}")
                add_screen_notification(f"Pupils: {status}", "info")
            
            elif key == ord('h'):
                show_hands = not show_hands
                status = "ON" if show_hands else "OFF"
                print(f"Hands: {status}")
                add_screen_notification(f"Hands: {status}", "info")
            
            elif key == ord('u'):
                show_ui_panel = not show_ui_panel
                status = "ON" if show_ui_panel else "OFF"
                print(f"UI panel: {status}")
                add_screen_notification(f"UI panel: {status}", "info")
            
            elif key == ord('s'):
                MODES[current_mode]["sound"] = not MODES[current_mode]["sound"]
                status = "ON" if MODES[current_mode]["sound"] else "OFF"
                print(f"Sound ({current_mode}): {status}")
                add_screen_notification(f"Sound: {status}", "info")
            
            elif key == ord('n'):
                enable_system_notifications = not enable_system_notifications
                status = "ON" if enable_system_notifications else "OFF"
                print(f"System notifications: {status}")
                add_screen_notification(f"System notifications: {status}", "info")
            
            elif key == ord('r'):
                distracted = False
                buffer_active = False
                eyes_closed_start_time = 0
                head_rotation_start_time = 0
                head_rotation_buffer_active = False
                played_return_sound = False
                print("State reset")
                add_screen_notification("State reset", "info")
            
            elif key == ord('z'):
                blink_count = 0
                blink_times.clear()
                print("Blink counter cleared")
                add_screen_notification("Blink counter cleared", "info")
                
            elif key == ord('c'):
                # Calibrate Relaxed
                calibration_active = True
                calibration_label = 0
                calibration_start = time.time()
                calibration_samples = 0
                load_classifier.feature_buffer.clear()
                add_screen_notification("Calibrating RELAXED (10s)", "info")
                
            elif key == ord('v'):
                # Calibrate Focused
                calibration_active = True
                calibration_label = 1
                calibration_start = time.time()
                calibration_samples = 0
                load_classifier.feature_buffer.clear()
                add_screen_notification("Calibrating FOCUSED (10s)", "info")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print(f"Final mode: {current_mode.upper()}")
    print(f"Overload level: {overload_level} ({overload_score}/100)")
    print(f"Total blinks: {blink_count}")
    print(f"Average FPS: {np.mean(fps_history):.1f}" if fps_history else "")
    print("=" * 70)

if __name__ == "__main__":
    main()