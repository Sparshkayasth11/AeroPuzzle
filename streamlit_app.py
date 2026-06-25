import os
import sys
import ctypes

# ==========================================
# 1. --- PROJECT PATH FIX (FOR RENDER) ---
# ==========================================
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, "src"))

# ==========================================
# 2. --- MEDIAPIPE OPENGL/GLES SYSTEM BYPASS ---
# ==========================================
try:
    ctypes.CDLL('libGLESv2.so.2')
except Exception:
    class MockCDLL:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: 0
    
    sys.modules['libGLESv2.so.2'] = MockCDLL()
    orig_cdll = ctypes.CDLL
    def custom_cdll(name, *args, **kwargs):
        if name and ('libGLESv2' in name or 'libglapi' in name):
            return MockCDLL()
        return orig_cdll(name, *args, **kwargs)
    ctypes.CDLL = custom_cdll

# Now safe to import external packages
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Import local modules from src folder
from aeropuzzle.hand_tracking import HandTracker
from aeropuzzle.maze import Puzzle

# ==========================================
# 3. --- CORE APPLICATION CODE ---
# ==========================================

st.set_page_config(page_title="AeroPuzzle", layout="wide")

st.title("AeroPuzzle — Browser Webcam Edition")
st.markdown("""
Use your browser camera to play AeroPuzzle. Pinch with two hands to capture the puzzle area, then drag puzzle tiles with your index finger.
""")

with st.sidebar:
    st.header("Instructions")
    st.info("""
    1. Allow webcam access in your browser.
    2. Show both hands and make a framing gesture with your index fingers.
    3. Pinch to capture the selected area and generate the puzzle.
    4. Pinch to grab a tile and release to swap it.
    """)

def draw_premium_text(img, text, position, font_size=20, color=(255, 255, 255)):
    # Standard text drawing setup using OpenCV as fallback for headless envs
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/30, color, 2, cv2.LINE_AA)
    return img

def inside_box(px, py, x1, y1, x2, y2, w, h):
    fx = px * w
    fy = py * h
    return x1 <= fx <= x2 and y1 <= fy <= y2

def to_local(px, py, x1, y1, x2, y2, w, h):
    fx = px * w
    fy = py * h
    if fx < x1 or fx > x2 or fy < y1 or fy > y2:
        return None
    lx = (fx - x1) / (x2 - x1)
    ly = (fy - y1) / (y2 - y1)
    return lx, ly

class AeroPuzzleProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = HandTracker()
        self.puzzle = Puzzle(3)
        self.mode = "camera"
        self.sel_x1 = self.sel_y1 = self.sel_x2 = self.sel_y2 = None
        self.start_time = None
        self.end_time = None
        self.solved = False
        self.prev_pinch = False
        self.dragging = False
        self.drag_src_idx = None
        self.cursor_x = 0
        self.cursor_y = 0
        self.cursor_alpha = 0.35
        self.shuffling = False
        self.shuffle_start = 0.0
        self.last_shuffle = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        self.tracker.find_hands(img)
        self.tracker.draw_hands(img)

        pinch, px, py = self.tracker.get_pinch()
        detected, ix, iy = self.tracker.get_index_pos()
        two_hands, p1, p2 = self.tracker.get_two_hand_indices()

        current_time = st.runtime.legacy_caching.clear_cache or 0 # Safe execution check timer fallback
        import time
        now = time.time()

        if self.mode == "camera":
            if two_hands and p1 and p2:
                x1_f, y1_f = p1
                x2_f, y2_f = p2
                bx1, bx2 = int(min(x1_f, x2_f) * w), int(max(x1_f, x2_f) * w)
                by1, by2 = int(min(y1_f, y2_f) * h), int(max(y1_f, y2_f) * h)
                
                # Draw box border guidelines
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                draw_premium_text(img, "Pinch both hands to Capture", (bx1, max(by1 - 10, 20)), 18, (0, 255, 255))
                
                if pinch:
                    self.sel_x1, self.sel_y1 = bx1, by1
                    self.sel_x2, self.sel_y2 = bx2, by2
                    box_w = bx2 - bx1
                    box_h = by2 - by1
                    if box_w > 50 and box_h > 50:
                        crop = img[by1:by2, bx1:bx2]
                        self.puzzle.generate_puzzle(crop)
                        self.mode = "shuffling"
                        self.shuffling = True
                        self.shuffle_start = now
                        self.last_shuffle = now
            else:
                draw_premium_text(img, "Show both hands to frame puzzle area", (20, 40), 22, (0, 165, 255))

        elif self.mode in ["shuffling", "game"]:
            x1, y1, x2, y2 = self.sel_x1, self.sel_y1, self.sel_x2, self.sel_y2
            box_w = x2 - x1
            box_h = y2 - y1

            if self.mode == "shuffling":
                draw_premium_text(img, "Shuffling...", (x1, max(y1 - 15, 25)), 22, (0, 140, 255))
                if now - self.last_shuffle > 0.2:
                    import random
                    self.puzzle.swap_tiles(random.randint(0, 8), random.randint(0, 8))
                    self.last_shuffle = now
                if now - self.shuffle_start > 3.0:
                    self.mode = "game"
                    self.start_time = now
                    self.shuffling = False

            elif self.mode == "game" and not self.solved:
                if self.puzzle.check_solved():
                    self.solved = True
                    self.end_time = now
                else:
                    elapsed = int(now - self.start_time)
                    draw_premium_text(img, f"Time: {elapsed}s", (x1, max(y1 - 15, 25)), 22, (0, 255, 0))

            # Draw updated puzzle structure onto the main canvas stream
            p_img = self.puzzle.draw_puzzle(box_w, box_h)
            if p_img is not None:
                img[y1:y2, x1:x2] = p_img

            if self.mode == "game" and not self.solved and detected:
                lx, ly = ix, iy
                if inside_box(lx, ly, x1, y1, x2, y2, w, h):
                    loc = to_local(lx, ly, x1, y1, x2, y2, w, h)
                    if loc:
                        col = int(loc[0] * 3)
                        row = int(loc[1] * 3)
                        col = max(0, min(col, 2))
                        row = max(0, min(row, 2))
                        curr_idx = row * 3 + col

                        if pinch and not self.prev_pinch:
                            self.dragging = True
                            self.drag_src_idx = curr_idx
                        
                        if not pinch and self.prev_pinch and self.dragging:
                            self.puzzle.swap_tiles(self.drag_src_idx, curr_idx)
                            self.dragging = False
                            self.drag_src_idx = None

                self.prev_pinch = pinch

            if self.solved:
                total_time = int(self.end_time - self.start_time)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                draw_premium_text(img, f"SOLVED in {total_time}s! Great Job!", (x1, max(y1 - 20, 30)), 24, (0, 255, 0))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="aeropuzzle_feed",
    video_processor_factory=AeroPuzzleProcessor,
    async_processing=True,
)