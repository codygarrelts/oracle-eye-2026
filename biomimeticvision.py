"""
BIOMIMETICVISION v2.6 — ORACLE EYE 2026 (Full Sheet)
Cody Garrelts — Dolphin-qualified ETV3  — Dec 2025
"""

import cv2
import numpy as np
import torch
import threading
import queue
import time
import math
import mss
import pyautogui
from collections import deque, defaultdict
from dataclasses import dataclass

# ==================== 2026 HYPER-MINMAXED CONFIG ====================
@dataclass
class OracleConfig:
    # Backbone
    YOLO_MODEL: str = "yolov11n.pt"           # drop-in model name
    TRACKER: str = "bytetrack.yaml"           # tracker config or path
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Oracle Soul — tuned
    MAX_HISTORY: int = 1024
    PRED_STEPS: int = 24
    BRANCHES: int = 7                         # +1 branch for extra prophecy
    GRAVITY: float = 0.982
    VELOCITY_SCALE: float = 1.55
    BASE_CONF: float = 0.31

    # Biomimetic Eye
    FOVEA_RADIUS: float = 0.15
    PERIPHERY_DECAY: float = 0.57
    SACCADE_THRESHOLD: float = 0.73
    ATTENTION_DECAY: float = 0.905
    IMPORTANCE_WEIGHTS: dict = None

    # Multi-modal vapors
    AUDIO_THRESH: float = 0.065
    DEPTH_ESTIMATION: bool = True

OracleConfig.IMPORTANCE_WEIGHTS = {
    "person": 1.0, "face": 1.7, "hand": 1.4, "car": 0.9, "crack": 2.1,
    "weld": 2.0, "tile": 1.9, "flap": 1.8, "green pencil": 2.5
}

# ==================== SOTA BACKBONE (drop-in) ====================
try:
    from ultralytics import YOLO
    model = YOLO(OracleConfig.YOLO_MODEL)
    model.to(OracleConfig.DEVICE)
    HAS_YOLO = True
    print(f"[Oracle] YOLO armed on {OracleConfig.DEVICE}")
except Exception as e:
    HAS_YOLO = False
    model = None
    print(f"[Oracle] No YOLO → entering apocalypse mode ({e})")

# Try to import BYTETracker if available (ultralytics tracker wrapper)
tracker = None
if HAS_YOLO:
    try:
        from ultralytics.trackers import BYTETracker
        tracker = BYTETracker(OracleConfig.TRACKER)
    except Exception:
        tracker = None

# ==================== APOCALYPSE-PROOF FALLBACK ====================
class DoomsdayDetector:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=500)
        self.id_counter = 0

    def detect(self, frame):
        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < 700: continue
            x, y, w, h = cv2.boundingRect(c)
            self.id_counter += 1
            boxes.append({
                "id": int(self.id_counter),
                "box": (x, y, x+w, y+h),
                "center": (x+w//2, y+h//2),
                "label": "anomaly",
                "score": float(min(0.96, cv2.contourArea(c)/20000))
            })
        return boxes

doomsday = DoomsdayDetector()

# ==================== YOUR SACRED ORACLE BRAIN (enhanced) ====================
class OracleBrain:
    def __init__(self):
        # memory: deque of dicts mapping id -> (x,y)
        self.memory = deque(maxlen=OracleConfig.MAX_HISTORY)
        self.soul_cache = {}
        # temporal influence weights per object id (exponential decay)
        self.temporal_weights = defaultdict(lambda: 1.0)
        # store latest known centers accessible by id
        self.latest = {}

    def _get_soul_seed(self, obj_id, patch):
        if obj_id not in self.soul_cache:
            try:
                if patch.size > 0:
                    flat = cv2.resize(patch, (64,64)).flatten()
                    self.soul_cache[obj_id] = hash(flat.tobytes()) % 2**32
                else:
                    self.soul_cache[obj_id] = np.random.randint(0, 2**32)
            except Exception:
                self.soul_cache[obj_id] = np.random.randint(0, 2**32)
        return self.soul_cache.get(obj_id, np.random.randint(0, 2**32))

    def physics_branch(self, obj):
        vx, vy = obj.get("velocity", (0.0, 0.0))
        vx *= OracleConfig.VELOCITY_SCALE
        vy *= OracleConfig.VELOCITY_SCALE
        x, y = float(obj["center"][0]), float(obj["center"][1])
        points = []
        for step in range(OracleConfig.PRED_STEPS):
            x += vx
            y += vy
            # gravity affects vertical velocity in a scaled manner
            vy += OracleConfig.GRAVITY * 0.09
            vx *= 0.988
            # probability shrink for far steps
            points.append((int(round(x)), int(round(y))))
        return {"points": points, "confidence": obj["score"] * 0.98}

    def behavior_branch(self, obj, patch):
        seed = self._get_soul_seed(obj["id"], patch)
        rng = np.random.default_rng(seed)
        x, y = float(obj["center"][0]), float(obj["center"][1])
        points = []
        angle = rng.uniform(0, 2*math.pi)
        speed = rng.uniform(3, 14)
        for _ in range(OracleConfig.PRED_STEPS):
            angle += rng.uniform(-0.35, 0.35)
            x += math.cos(angle) * speed
            y += math.sin(angle) * speed
            points.append((int(round(x)), int(round(y))))
            speed *= 0.995
        return {"points": points, "confidence": obj["score"] * 0.91}

    def probabilistic_cone(self, branches):
        # Create a probabilistic occupancy map from branches
        # For efficiency, sample points and aggregate into heatmap-like dict
        heat = defaultdict(float)
        for b in branches:
            conf = b.get("confidence", 0.5)
            pts = b.get("points", [])
            for idx, p in enumerate(pts):
                # weight earlier steps slightly more
                heat[p] += conf * (1.0 / (1 + idx*0.06))
        # generate sorted list of positions with scores
        items = sorted(heat.items(), key=lambda t: -t[1])
        cone = [pos for pos,score in items[:64]]  # top-n probable points
        return cone

    def prophesy(self, obj, patch):
        # compute branches
        branches = [self.physics_branch(obj)]
        branches += [self.behavior_branch(obj, patch) for _ in range(OracleConfig.BRANCHES-1)]

        # choose best by confidence
        best = max(branches, key=lambda b: b.get("confidence", 0.0))

        # compute probabilistic cone
        cone = self.probabilistic_cone(branches)

        # update attention (temporal smoothing)
        att_prev = self.temporal_weights.get(obj["id"], 1.0)
        att = max(0.0, min(1.0, best.get("confidence", 0.0)))
        # smooth update
        self.temporal_weights[obj["id"]] = att_prev * 0.85 + att * 0.15

        # prophecy failure if everything is low confidence
        prophecy_failure = all(b.get("confidence", 0.0) < 0.34 for b in branches)

        return {
            "id": obj["id"],
            "branches": branches,
            "best": best.get("points", []),
            "cone": cone,
            "attention": self.temporal_weights[obj["id"]],
            "prophecy_failure": prophecy_failure
        }

oracle = OracleBrain()

# ==================== WATCH-DOGS TRANSPARENT OVERLAY (polished) ====================
def draw_hud(frame, predictions):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Fovea
    cx, cy = w//2, h//2
    cv2.circle(overlay, (cx, cy), int(OracleConfig.FOVEA_RADIUS * min(w,h)), (0,255,255), 2)

    # Trails (subtle)
    for p in predictions:
        # draw all branch trails faintly
        for b in p.get("branches", []):
            pts = b.get("points", [])
            for i in range(1, len(pts)):
                cv2.line(overlay, pts[i-1], pts[i], (160,160,220), 1)
        # Draw best future in neon
        best = p.get("best", [])
        for i in range(1, len(best)):
            cv2.line(overlay, best[i-1], best[i], (0,220,140), 3)
        # Draw probabilistic cone as soft points
        for idx, c in enumerate(p.get("cone", [])[:40]):
            alpha = 0.9 * (1 - idx/40)
            cv2.circle(overlay, c, max(1, int(3*(1-alpha))), (0,180,255), -1)

        # attention meter per object
        att = p.get("attention", 0.0)
        label = f"ID:{p.get('id',0)} A:{att:.2f}"
        # place label near start of best if available
        if best:
            x,y = best[0]
            cv2.putText(overlay, label, (max(10,x-20), max(20,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Prophecy scream (less invasive)
    if any(p.get("prophecy_failure") for p in predictions):
        cv2.putText(overlay, "PROPHECY FAILURE — REALITY LIES", (40, 80), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0,0,255), 3, cv2.LINE_AA)

    # HUD blend
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    return frame

# ==================== DEPTH ESTIMATION PLACEHOLDER (simple heuristics) ====================
# Real depth models would require extra dependencies; here we provide a safe placeholder
# that uses box size as a proxy for depth (larger box => closer). If the user has
# an off-the-shelf monocular depth model, they can replace `estimate_depth` logic.

def estimate_depth(box, frame_shape):
    x1,y1,x2,y2 = box
    w = x2 - x1
    h = y2 - y1
    frame_w, frame_h = frame_shape[1], frame_shape[0]
    # normalized area inverse => depth approximation
    area_norm = (w*h) / float(frame_w*frame_h + 1e-8)
    # clamp and invert so bigger area => smaller depth value
    depth_score = max(0.01, min(10.0, 1.0 / (area_norm + 1e-4)))
    return depth_score

# ==================== MAIN LOOP — YOUR STRUCTURE, patched ====================

def run_oracle(source="0"):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    print("[Oracle] Awakened. Dreaming 7 futures per soul…")

    with mss.mss() as sct:
        while True:
            ret, frame = cap.read()
            if not ret:
                if source == "screen":
                    frame = np.array(sct.grab(sct.monitors[1]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    break

            frame = cv2.resize(frame, (1280, 720))

            predictions = []

            # ------------------- DETECTION + ID HANDLING -------------------
            if HAS_YOLO and model is not None and tracker is not None:
                try:
                    results = model.track(frame, persist=True, tracker=OracleConfig.TRACKER, verbose=False)[0]
                    dets = results.boxes

                    if dets is not None and len(dets) > 0:
                        xyxy = dets.xyxy.cpu().numpy()
                        confs = dets.conf.cpu().numpy()
                        clss  = dets.cls.cpu().numpy()
                        # some tracker wrappers put ids in dets.id, else fallback
                        ids = None
                        try:
                            ids = dets.id.cpu().numpy()
                        except Exception:
                            ids = None

                        if ids is None:
                            ids = np.arange(len(dets))

                        for (x1,y1,x2,y2), conf, cls_id, obj_id in zip(xyxy, confs, clss, ids):
                            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                            patch = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                            cx, cy = int((x1+x2)//2), int((y1+y2)//2)

                            obj = {
                                "id": int(obj_id),
                                "center": (cx, cy),
                                "box": (x1, y1, x2, y2),
                                "score": float(conf),
                            }

                            # velocity from memory
                            prev_center = oracle.latest.get(obj["id"])
                            if prev_center is not None:
                                vx = obj["center"][0] - prev_center[0]
                                vy = obj["center"][1] - prev_center[1]
                                obj["velocity"] = (float(vx), float(vy))
                            else:
                                obj["velocity"] = (0.0, 0.0)

                            # update latest and memory
                            oracle.latest[obj["id"]] = obj["center"]
                            oracle.memory.append({obj["id"]: obj["center"]})

                            # depth heuristic
                            if OracleConfig.DEPTH_ESTIMATION:
                                obj["depth"] = estimate_depth(obj["box"], frame.shape)

                            predictions.append(oracle.prophesy(obj, patch))
                except Exception as e:
                    # degrade gracefully to doomsday
                    print(f"[Oracle] Tracker/YOLO failed: {e}")
                    detections = doomsday.detect(frame)
                    for d in detections:
                        x1,y1,x2,y2 = d["box"]
                        patch = frame[y1:y2, x1:x2]
                        obj = {"id": d["id"], "center": d["center"], "box": d["box"], "score": d["score"], "velocity": (0,0)}
                        oracle.latest[obj["id"]] = obj["center"]
                        oracle.memory.append({obj["id"]: obj["center"]})
                        if OracleConfig.DEPTH_ESTIMATION:
                            obj["depth"] = estimate_depth(obj["box"], frame.shape)
                        predictions.append(oracle.prophesy(obj, patch))

            else:
                # APOCALYPSE MODE
                detections = doomsday.detect(frame)
                for d in detections:
                    x1,y1,x2,y2 = d["box"]
                    patch = frame[y1:y2, x1:x2]
                    obj = {"id": d["id"], "center": d["center"], "box": d["box"], "score": d["score"], "velocity": (0,0)}
                    oracle.latest[obj["id"]] = obj["center"]
                    oracle.memory.append({obj["id"]: obj["center"]})
                    if OracleConfig.DEPTH_ESTIMATION:
                        obj["depth"] = estimate_depth(obj["box"], frame.shape)
                    predictions.append(oracle.prophesy(obj, patch))

            # ------------------- RENDER -------------------
            frame = draw_hud(frame, predictions)
            cv2.imshow("Oracle Eye v2.6 — 2026", frame)

            k = cv2.waitKey(1)
            if k == 27:  # ESC to exit
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", default="0", help="0=webcam | path=video | screen")
    args = p.parse_args()
    run_oracle(args.source)
