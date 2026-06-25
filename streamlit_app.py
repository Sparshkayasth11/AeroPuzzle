"""
AeroPuzzle — Web Edition
========================
Streamlit + streamlit-webrtc wrapper that converts the desktop OpenCV game
into a live browser-based experience. All game logic (hand tracking, puzzle,
drag-and-drop) runs inside a VideoProcessorBase.recv() callback.
"""

import sys
import os

# ── Make the src/ package importable without pip install ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
import av
import cv2
import time
import random
import numpy as np

from aeropuzzle.hand_tracking import HandTracker
from aeropuzzle.maze import Puzzle
from aeropuzzle.app import draw_panel, draw_premium_text, inside_box, to_local, draw_grid


# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AeroPuzzle — Gesture-Controlled Puzzle",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ---- Dark premium background ---- */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
    }

    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid #1e293b;
    }

    /* ---- Header glow ---- */
    .aero-title {
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #00f2fe);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 200% center; }
    }

    .aero-subtitle {
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #64748b;
        text-align: center;
        margin-top: -8px;
        margin-bottom: 24px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    /* ---- Instruction cards ---- */
    .step-card {
        background: linear-gradient(145deg, #111827, #1a1f2e);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: border-color 0.3s ease;
    }
    .step-card:hover {
        border-color: #00f2fe;
    }
    .step-card .step-num {
        color: #00f2fe;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.1em;
    }
    .step-card .step-text {
        color: #e2e8f0;
        font-size: 0.95rem;
        margin-top: 4px;
        line-height: 1.5;
    }

    /* ---- Status badge ---- */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .status-live {
        background: rgba(0, 242, 254, 0.12);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.3);
    }

    /* ---- WebRTC container styling ---- */
    .stVideo, [data-testid="stVideo"] {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid #1e293b;
        box-shadow: 0 0 40px rgba(0, 242, 254, 0.06);
    }

    /* ---- Footer ---- */
    .footer-text {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid #1e293b;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════
#  SIDEBAR — How to Play
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="aero-title" style="font-size:1.8rem;">✋ AeroPuzzle</p>', unsafe_allow_html=True)
    st.markdown('<p class="aero-subtitle" style="font-size:0.75rem;">Web Edition</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎮 How to Play")

    steps = [
        ("STEP 01", "Show **both hands** to the camera. Use your **index fingers** to frame the area you want to capture."),
        ("STEP 02", "**Pinch** (thumb + index) to capture the framed area as a 3×3 sliding puzzle."),
        ("STEP 03", "Move your hand to control the cursor. **Pinch to grab** a puzzle tile."),
        ("STEP 04", "**Drag** the tile to another position and **release** to swap them."),
        ("STEP 05", "Solve the puzzle! Your time is tracked. The game **auto-resets** after you win."),
    ]

    for num, text in steps:
        st.markdown(
            f'<div class="step-card">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-text">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚙️ Tips")
    st.markdown(
        """
        - Use a **well-lit** environment
        - Keep hands **within frame**
        - Works best on **Chrome / Edge**
        - Allow **camera access** when prompted
        """,
    )

    st.markdown(
        '<div class="footer-text">'
        'Built with OpenCV · MediaPipe · Streamlit<br>'
        '<a href="https://github.com/Sparshkayasth11/AeroPuzzle" target="_blank" '
        'style="color:#00f2fe;">GitHub ↗</a>'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
#  VIDEO PROCESSOR — Core Game Logic
# ═══════════════════════════════════════════════════════════
class AeroPuzzleProcessor(VideoProcessorBase):
    """Processes each webcam frame through the AeroPuzzle game pipeline.
    Mirrors the logic from aeropuzzle.app.main() but runs per-frame
    inside streamlit-webrtc instead of a cv2.VideoCapture loop."""

    def __init__(self):
        # Lazy-init tracker (MediaPipe loads on first frame)
        self._tracker: HandTracker | None = None
        self.puzzle = Puzzle(3)

        # Game mode: "camera" or "puzzle"
        self.mode = "camera"

        # Selection box (pixel coordinates in the captured frame)
        self.sel_x1 = self.sel_y1 = self.sel_x2 = self.sel_y2 = None

        # Game state
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.solved = False

        # Pinch / drag state
        self.prev_pinch = False
        self.dragging = False
        self.drag_src_idx: int | None = None

        # Cursor (pixel coords)
        self.cursor_x = 0
        self.cursor_y = 0
        self.cursor_alpha = 0.35  # smoothing for hover

        # Shuffle animation
        self.shuffling = False
        self.shuffle_start: float = 0
        self.last_shuffle: float = 0

    @property
    def tracker(self) -> HandTracker:
        if self._tracker is None:
            self._tracker = HandTracker()
        return self._tracker

    # ------------------------------------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # ── Run hand detection ──
        self.tracker.find_hands(img)
        self.tracker.draw_hands(img)

        pinch, px, py = self.tracker.get_pinch()
        detected, ix, iy = self.tracker.get_index_pos()
        two_hands, p1, p2 = self.tracker.get_two_hand_indices()

        h, w, _ = img.shape

        # ═════════════ CAMERA MODE ═════════════
        if self.mode == "camera":
            output = self._camera_mode(img, pinch, two_hands, p1, p2, w, h)
        # ═════════════ PUZZLE MODE ═════════════
        else:
            output = self._puzzle_mode(
                img, pinch, px, py, detected, ix, iy, w, h,
            )

        self.prev_pinch = pinch
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    # ------------------------------------------------------------------
    #  CAMERA MODE
    # ------------------------------------------------------------------
    def _camera_mode(self, frame, pinch, two_hands, p1, p2, w, h):
        # Sleek white border
        bm = 20
        cv2.rectangle(frame, (bm, bm), (w - bm, h - bm), (255, 255, 255), 6)

        # Header box
        tw_, th_ = 320, 60
        tx = w // 2 - tw_ // 2
        ty = bm
        cv2.rectangle(frame, (tx, ty), (tx + tw_, ty + th_), (255, 255, 255), -1)
        frame = draw_premium_text(
            frame, "AeroPuzzle", (tx + 65, ty + 8),
            font_size=32, color=(15, 17, 26), bold=True,
        )

        # Status pill
        status_text = "TWO HANDS NOT DETECTED"
        status_color = (127, 0, 255)
        if two_hands:
            status_text = "PINCH TO SNAP & START"
            status_color = (160, 245, 0)

        draw_panel(
            frame, w - 380, 32, 340, 45,
            bg_color=(30, 25, 20), opacity=0.8,
            border_color=status_color, border_thickness=1,
        )
        frame = draw_premium_text(
            frame, status_text, (w - 360, 43),
            font_size=16, color=status_color, bold=True,
        )

        # Two-hand rectangle
        if two_hands:
            x1, y1 = int(p1[0] * w), int(p1[1] * h)
            x2, y2 = int(p2[0] * w), int(p2[1] * h)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            color = (254, 242, 0)
            if pinch:
                color = (127, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Pinch to capture
            if pinch and not self.prev_pinch:
                if abs(x2 - x1) > 100 and abs(y2 - y1) > 100:
                    self.sel_x1, self.sel_y1 = x1, y1
                    self.sel_x2, self.sel_y2 = x2, y2
                    crop = frame[self.sel_y1:self.sel_y2, self.sel_x1:self.sel_x2]
                    if crop.size != 0:
                        self.puzzle.create(crop)
                        self.shuffling = True
                        self.shuffle_start = time.time()
                        self.last_shuffle = 0
                        self.mode = "puzzle"
                        self.start_time = None
                        self.solved = False
                        self.end_time = None

        # Bottom hint
        draw_panel(
            frame, w // 2 - 250, h - 60, 500, 40,
            bg_color=(20, 16, 13), opacity=0.8,
            border_color=(255, 255, 255), border_thickness=1,
        )
        frame = draw_premium_text(
            frame,
            "Use index fingers of both hands to frame your photo.",
            (w // 2 - 210, h - 50), font_size=14, color=(200, 200, 200),
        )

        return frame

    # ------------------------------------------------------------------
    #  PUZZLE MODE
    # ------------------------------------------------------------------
    def _puzzle_mode(self, frame, pinch, px, py, detected, ix, iy, w, h):
        output = frame.copy()

        # Overlay puzzle on the selection region
        if self.sel_x1 is not None:
            puzzle_img = self.puzzle.combine()
            puzzle_img = cv2.resize(
                puzzle_img,
                (self.sel_x2 - self.sel_x1, self.sel_y2 - self.sel_y1),
            )
            output[self.sel_y1:self.sel_y2, self.sel_x1:self.sel_x2] = puzzle_img
            draw_grid(output, self.sel_x1, self.sel_y1, self.sel_x2, self.sel_y2)

        # ── Cursor position ──
        if pinch and detected:
            self.cursor_x = int(px * w)
            self.cursor_y = int(py * h)
        elif detected:
            raw_cx, raw_cy = int(ix * w), int(iy * h)
            self.cursor_x = int(
                self.cursor_alpha * raw_cx + (1 - self.cursor_alpha) * self.cursor_x
            )
            self.cursor_y = int(
                self.cursor_alpha * raw_cy + (1 - self.cursor_alpha) * self.cursor_y
            )

        # Clamp cursor to puzzle area
        if self.sel_x1 is not None:
            self.cursor_x = max(self.sel_x1, min(self.cursor_x, self.sel_x2))
            self.cursor_y = max(self.sel_y1, min(self.cursor_y, self.sel_y2))

        cursor_nx = self.cursor_x / w if w else 0
        cursor_ny = self.cursor_y / h if h else 0

        # ── Shuffle animation ──
        if self.shuffling:
            now = time.time()
            if now - self.shuffle_start < 1.8:
                if now - self.last_shuffle > 0.05:
                    i = random.randint(0, len(self.puzzle.tiles) - 1)
                    j = random.randint(0, len(self.puzzle.tiles) - 1)
                    self.puzzle.swap(i, j)
                    self.last_shuffle = now
            else:
                self.shuffling = False
                self.start_time = time.time()

        # ── Interaction ──
        if not self.shuffling and detected:
            if pinch and not self.prev_pinch:
                # Pinch start: grab a tile
                if inside_box(
                    cursor_nx, cursor_ny,
                    self.sel_x1, self.sel_y1, self.sel_x2, self.sel_y2, w, h,
                ):
                    local = to_local(
                        cursor_nx, cursor_ny,
                        self.sel_x1, self.sel_y1, self.sel_x2, self.sel_y2, w, h,
                    )
                    if local is not None:
                        lx, ly = local
                        idx = self.puzzle.get_index(lx, ly)
                        if idx is not None:
                            self.drag_src_idx = idx
                            self.puzzle.selected = idx
                            self.dragging = True

            elif not pinch and self.prev_pinch:
                # Pinch release: drop the tile
                if self.dragging and self.drag_src_idx is not None:
                    local = to_local(
                        cursor_nx, cursor_ny,
                        self.sel_x1, self.sel_y1, self.sel_x2, self.sel_y2, w, h,
                    )
                    if local is not None:
                        lx, ly = local
                        idx2 = self.puzzle.get_index(lx, ly)
                        if idx2 is not None:
                            self.puzzle.swap(self.drag_src_idx, idx2)

                    self.puzzle.selected = None
                    self.drag_src_idx = None
                    self.dragging = False

                    if not self.solved and self.puzzle.is_solved(self.puzzle.original_tiles):
                        self.solved = True
                        self.end_time = time.time()

        # ── Draw drag visuals ──
        gs = self.puzzle.grid_size
        cell_w = (self.sel_x2 - self.sel_x1) // gs if self.sel_x1 is not None else 0
        cell_h = (self.sel_y2 - self.sel_y1) // gs if self.sel_y1 is not None else 0

        if self.dragging and detected and self.sel_x1 is not None:
            local = to_local(
                cursor_nx, cursor_ny,
                self.sel_x1, self.sel_y1, self.sel_x2, self.sel_y2, w, h,
            )
            if local is not None:
                lx, ly = local
                hover_idx = self.puzzle.get_index(lx, ly)
                if hover_idx is not None and hover_idx != self.drag_src_idx:
                    hr, hc = hover_idx // gs, hover_idx % gs
                    hx1 = self.sel_x1 + hc * cell_w
                    hy1 = self.sel_y1 + hr * cell_h
                    hx2, hy2 = hx1 + cell_w, hy1 + cell_h
                    overlay = output.copy()
                    cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (254, 242, 0), -1)
                    cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
                    cv2.rectangle(output, (hx1, hy1), (hx2, hy2), (254, 242, 0), 2)

        # Highlight source tile
        if self.drag_src_idx is not None and cell_w > 0:
            sr, sc = self.drag_src_idx // gs, self.drag_src_idx % gs
            sx1 = self.sel_x1 + sc * cell_w
            sy1 = self.sel_y1 + sr * cell_h
            sx2, sy2 = sx1 + cell_w, sy1 + cell_h
            cv2.rectangle(output, (sx1, sy1), (sx2, sy2), (127, 0, 255), 3)

            # Ghost tile following cursor
            tile_img = self.puzzle.tiles[self.drag_src_idx]
            ghost_w, ghost_h = cell_w // 2, cell_h // 2
            if ghost_w > 0 and ghost_h > 0:
                ghost = cv2.resize(tile_img, (ghost_w, ghost_h))
                gx1 = self.cursor_x - ghost_w // 2
                gy1 = self.cursor_y - ghost_h // 2
                gx2, gy2 = gx1 + ghost_w, gy1 + ghost_h

                src_x1, src_y1 = max(0, -gx1), max(0, -gy1)
                dst_x1, dst_y1 = max(0, gx1), max(0, gy1)
                dst_x2, dst_y2 = min(w, gx2), min(h, gy2)
                src_x2 = src_x1 + (dst_x2 - dst_x1)
                src_y2 = src_y1 + (dst_y2 - dst_y1)

                if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                    ghost_region = ghost[src_y1:src_y2, src_x1:src_x2]
                    frame_region = output[dst_y1:dst_y2, dst_x1:dst_x2]
                    blended = cv2.addWeighted(ghost_region, 0.7, frame_region, 0.3, 0)
                    output[dst_y1:dst_y2, dst_x1:dst_x2] = blended

        # ── Cursor ──
        if detected:
            if self.dragging:
                cv2.circle(output, (self.cursor_x, self.cursor_y), 10, (127, 0, 255), -1)
                cv2.circle(output, (self.cursor_x, self.cursor_y), 12, (255, 255, 255), 2)
            else:
                cv2.circle(output, (self.cursor_x, self.cursor_y), 7, (255, 255, 255), -1)
                cv2.circle(output, (self.cursor_x, self.cursor_y), 9, (254, 242, 0), 2)

        # ── HUD ──
        bm = 20
        cv2.rectangle(output, (bm, bm), (w - bm, h - bm), (255, 255, 255), 6)

        tw_, th_ = 320, 60
        tx = w // 2 - tw_ // 2
        ty = bm
        cv2.rectangle(output, (tx, ty), (tx + tw_, ty + th_), (255, 255, 255), -1)
        output = draw_premium_text(
            output, "AeroPuzzle", (tx + 65, ty + 8),
            font_size=32, color=(15, 17, 26), bold=True,
        )

        if self.start_time and not self.solved:
            elapsed = time.time() - self.start_time
            output = draw_premium_text(
                output, f"TIME: {elapsed:.2f}s", (w - 200, 38),
                font_size=20, color=(160, 245, 0), bold=True,
            )
            draw_panel(
                output, w // 2 - 200, h - 60, 400, 40,
                bg_color=(20, 16, 13), opacity=0.8,
                border_color=(255, 255, 255), border_thickness=1,
            )
            output = draw_premium_text(
                output,
                "Pinch to grab a tile. Move & release to swap.",
                (w // 2 - 170, h - 50), font_size=14, color=(220, 220, 220),
            )

        if self.solved and self.end_time and self.start_time:
            final_time = self.end_time - self.start_time
            v_w, v_h = 420, 220
            v_x = w // 2 - v_w // 2
            v_y = h // 2 - v_h // 2

            draw_panel(
                output, v_x, v_y, v_w, v_h,
                bg_color=(15, 24, 16), opacity=0.85,
                border_color=(160, 245, 0), border_thickness=2,
            )
            output = draw_premium_text(
                output, "PUZZLE SOLVED!", (v_x + 95, v_y + 40),
                font_size=32, color=(160, 245, 0), bold=True,
            )
            output = draw_premium_text(
                output, f"Final Time: {final_time:.2f} seconds",
                (v_x + 90, v_y + 110), font_size=20, color=(255, 255, 255),
            )

            time_left = max(0.0, 5.0 - (time.time() - self.end_time))
            output = draw_premium_text(
                output, f"Restarting in {time_left:.1f}s...",
                (v_x + 120, v_y + 165), font_size=14, color=(160, 160, 160),
            )

            # Auto-reset to camera mode after 5 seconds
            if time.time() - self.end_time > 5.0:
                self.mode = "camera"
                self.solved = False
                self.start_time = None
                self.end_time = None
                self.puzzle = Puzzle(3)
                self.sel_x1 = self.sel_y1 = self.sel_x2 = self.sel_y2 = None
                self.dragging = False
                self.drag_src_idx = None

        return output


# ═══════════════════════════════════════════════════════════
#  MAIN UI
# ═══════════════════════════════════════════════════════════
st.markdown('<p class="aero-title">AeroPuzzle</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="aero-subtitle">Gesture-Controlled Sliding Puzzle</p>',
    unsafe_allow_html=True,
)

# Centered columns for the video stream
col_l, col_c, col_r = st.columns([1, 6, 1])

with col_c:
    st.markdown(
        '<span class="status-badge status-live">● LIVE</span>',
        unsafe_allow_html=True,
    )

    # Use STUN servers to help establish peer-to-peer connections for client webcams
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
        ]
    })

    ctx = webrtc_streamer(
        key="aeropuzzle",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AeroPuzzleProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        video_html_attrs={
            "video": True,
            "audio": False,
            "style": {"width": "100%"},
        },
        async_processing=True,
    )

    if ctx and getattr(ctx, 'state', None) and getattr(ctx.state, 'playing', False):
        st.caption("🟢 Camera active — show your hands to begin!")
    else:
        st.info(
            "👆 Click **START** above to enable your camera and begin playing.",
            icon="📷",
        )

st.markdown(
    '<div class="footer-text">'
    "AeroPuzzle © 2025 · Built with ❤️ using OpenCV, MediaPipe & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
