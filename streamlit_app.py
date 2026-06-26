import os
import sys
import time
import random

# Handshake manual extracted dynamic libraries path injection
root_dir = os.path.dirname(os.path.abspath(__file__))
apt_lib_path = os.path.join(root_dir, ".apt", "usr", "lib", "x86_64-linux-gnu")
if os.path.exists(apt_lib_path):
    os.environ["LD_LIBRARY_PATH"] = apt_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
# Standard folder paths routing structure resolution
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
src_path = os.path.join(root_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Safe modules standard execution imports
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Safe Layered Core Fallback System Layout
try:
    from aeropuzzle.hand_tracking import HandTracker
    from aeropuzzle.maze import Puzzle
except (ModuleNotFoundError, ImportError):
    from src.aeropuzzle.hand_tracking import HandTracker
    from src.aeropuzzle.maze import Puzzle

# ==========================================
# CORE APPLICATION CODE
# ==========================================

st.set_page_config(page_title="AeroPuzzle", layout="wide")

st.title("AeroPuzzle — Browser Webcam Edition")
st.markdown("""
Use your browser camera to play AeroPuzzle. Bring both hands closer and make a framing gesture with your index fingers to select the crop box, then pinch to drag tiles!
""")

with st.sidebar:
    st.header("Instructions")
    st.info("""
    1. Allow webcam access in your browser.
    2. Show both hands and make a framing gesture with your index fingers.
    3. Hold the frame still for 1.5 seconds to auto-capture and shuffle the puzzle.
    4. Once the puzzle loads, pinch your thumb and index finger to grab and drag tiles.
    """)

def draw_premium_text(img, text, position, font_size=20, color=(255, 255, 255)):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/30, color, 2, cv2.LINE_AA)
    return img

@st.cache_resource
def load_instances():
    return HandTracker(), Puzzle(3)

_GLOBAL_TRACKER, _GLOBAL_PUZZLE = load_instances()

class AeroPuzzleProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = _GLOBAL_TRACKER
        self.puzzle = _GLOBAL_PUZZLE
        self.mode = "camera"
        self.sel_x1 = self.sel_y1 = self.sel_x2 = self.sel_y2 = None
        self.start_time = None
        self.end_time = None
        self.solved = False
        self.dragging = False
        self.drag_src_idx = None
        self.shuffling = False
        self.shuffle_start = 0.0
        self.last_shuffle = 0.0
        
        # Tracking hold states with built-in flicker protection
        self.frame_hold_start = None
        self.last_seen_hands = None
        self.cached_box = None
        
        # Local Grid System to bypass internal engine black screens
        self.local_tiles = []
        self.tile_indices = list(range(9))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        try:
            self.tracker.find_hands(img)
            self.tracker.draw_hands(img)
            
            pinch, px, py = self.tracker.get_pinch()
            detected, ix, iy = self.tracker.get_index_pos()
            two_hands, p1, p2 = self.tracker.get_two_hand_indices()
        except Exception:
            pinch, detected, two_hands = False, False, False
            px, py, ix, iy, p1, p2 = 0, 0, 0, 0, None, None

        now = time.time()

        if self.mode == "camera":
            if two_hands and p1 and p2:
                self.last_seen_hands = now
                x1_f, y1_f = p1
                x2_f, y2_f = p2
                bx1, bx2 = int(min(x1_f, x2_f) * w), int(max(x1_f, x2_f) * w)
                by1, by2 = int(min(y1_f, y2_f) * h), int(max(y1_f, y2_f) * h)
                
                # Boundaries safe containment check
                bx1, bx2 = max(0, min(bx1, w-1)), max(0, min(bx2, w-1))
                by1, by2 = max(0, min(by1, h-1)), max(0, min(by2, h-1))
                self.cached_box = (bx1, by1, bx2, by2)
            
            if self.last_seen_hands is not None and (now - self.last_seen_hands < 0.6) and self.cached_box is not None:
                bx1, by1, bx2, by2 = self.cached_box

                if self.frame_hold_start is None:
                    self.frame_hold_start = now
                
                elapsed_hold = now - self.frame_hold_start
                remaining = max(0.0, 1.5 - elapsed_hold)
                
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                if remaining > 0:
                    draw_premium_text(img, f"Hold Still to Capture... {remaining:.1f}s", (bx1, max(by1 - 10, 20)), 18, (0, 255, 255))
                
                if remaining <= 0:
                    self.sel_x1, self.sel_y1 = bx1, by1
                    self.sel_x2, self.sel_y2 = bx2, by2
                    box_w = bx2 - bx1
                    box_h = by2 - by1
                    
                    crop = img[by1:by2, bx1:bx2].copy()
                    if crop is None or crop.size == 0 or box_w < 20 or box_h < 20:
                        crop = np.zeros((300, 300, 3), dtype=np.uint8)
                        cv2.putText(crop, "AeroPuzzle", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    
                    crop_resized = cv2.resize(crop, (300, 300))
                    
                    # Pre-slice 3x3 array 
                    self.local_tiles = []
                    tw, th = 100, 100
                    for r in range(3):
                        for c in range(3):
                            tile_slice = crop_resized[r*th:(r+1)*th, c*tw:(c+1)*tw].copy()
                            cv2.rectangle(tile_slice, (0,0), (tw, th), (100, 100, 100), 1)
                            self.local_tiles.append(tile_slice)
                    
                    self.tile_indices = list(range(9))
                    self.mode = "shuffling"
                    self.shuffling = True
                    self.shuffle_start = now
                    self.last_shuffle = now
                    
                    try:
                        self.puzzle.generate_puzzle(crop_resized)
                    except Exception:
                        pass
                    
                    self.frame_hold_start = None
                    self.last_seen_hands = None
            else:
                self.frame_hold_start = None
                draw_premium_text(img, "Show both hands to frame puzzle area", (20, 40), 22, (0, 165, 255))

        elif self.mode in ["shuffling", "game"]:
            x1, y1, x2, y2 = self.sel_x1, self.sel_y1, self.sel_x2, self.sel_y2
            box_w, box_h = x2 - x1, y2 - y1

            if self.mode == "shuffling":
                draw_premium_text(img, "Shuffling Tiles...", (x1, max(y1 - 15, 25)), 22, (0, 140, 255))
                if now - self.last_shuffle > 0.15:
                    idx1, idx2 = random.randint(0, 8), random.randint(0, 8)
                    self.tile_indices[idx1], self.tile_indices[idx2] = self.tile_indices[idx2], self.tile_indices[idx1]
                    try:
                        self.puzzle.swap_tiles(idx1, idx2)
                    except Exception:
                        pass
                    self.last_shuffle = now
                    
                if now - self.shuffle_start > 2.0:
                    if self.tile_indices == list(range(9)):
                        random.shuffle(self.tile_indices)
                    self.mode = "game"
                    self.start_time = now
                    self.shuffling = False

            elif self.mode == "game" and not self.solved:
                if self.tile_indices == list(range(9)):
                    self.solved = True
                    self.end_time = now
                else:
                    elapsed = int(now - self.start_time)
                    draw_premium_text(img, f"Time: {elapsed}s", (x1, max(y1 - 15, 25)), 22, (0, 255, 0))

            # RENDER ENGINE
            try:
                p_img = None
                try:
                    p_img = self.puzzle.draw_puzzle(box_w, box_h)
                except Exception:
                    pass
                
                if p_img is None or p_img.size == 0 or np.all(p_img == 0):
                    backup_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
                    tw, th = 100, 100
                    for i, exact_pos in enumerate(self.tile_indices):
                        r, c = i // 3, i % 3
                        backup_canvas[r*th:(r+1)*th, c*tw:(c+1)*tw] = self.local_tiles[exact_pos]
                    p_img = cv2.resize(backup_canvas, (box_w, box_h))
                
                if p_img is not None and p_img.size > 0:
                    img[y1:y2, x1:x2] = p_img
            except Exception:
                pass

            # DRAG-AND-DROP INSTANT REGISTRATION 
            if self.mode == "game" and not self.solved and detected:
                finger_x = int(ix * w) if ix <= 1.0 else int(ix)
                finger_y = int(iy * h) if iy <= 1.0 else int(iy)
                
                if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                    cv2.circle(img, (finger_x, finger_y), 6, (255, 0, 255), -1)
                    
                    local_x = finger_x - x1
                    local_y = finger_y - y1
                    
                    col = max(0, min(int((local_x / box_w) * 3), 2))
                    row = max(0, min(int((local_y / box_h) * 3), 2))
                    curr_idx = row * 3 + col

                    if pinch:
                        if not self.dragging:
                            self.dragging = True
                            self.drag_src_idx = curr_idx
                        else:
                            # Instant responsive shift if index tracking changes during active drag
                            if self.drag_src_idx is not None and self.drag_src_idx != curr_idx:
                                s_idx = self.drag_src_idx
                                self.tile_indices[s_idx], self.tile_indices[curr_idx] = self.tile_indices[curr_idx], self.tile_indices[s_idx]
                                try:
                                    self.puzzle.swap_tiles(s_idx, curr_idx)
                                except Exception:
                                    pass
                                self.drag_src_idx = curr_idx # Move anchor to current position
                        
                        cv2.rectangle(img, (x1 + col*int(box_w/3), y1 + row*int(box_h/3)), 
                                      (x1 + (col+1)*int(box_w/3), y1 + (row+1)*int(box_h/3)), (0, 255, 0), 2)
                    else:
                        self.dragging = False
                        self.drag_src_idx = None

            if self.solved:
                total_time = int(self.end_time - self.start_time)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                draw_premium_text(img, f"SOLVED in {total_time}s! Brilliant!", (x1, max(y1 - 20, 30)), 24, (0, 255, 0))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]
}

webrtc_streamer(
    key="aeropuzzle_feed",
    video_processor_factory=AeroPuzzleProcessor,
    async_processing=True,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 20}},
        "audio": False
    }
)