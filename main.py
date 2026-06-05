#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
import argparse
import sys
from collections import deque

try:
    from serial.tools import list_ports
except Exception:
    list_ports = None

# ==========================
# Helper: find serial ports
# ==========================
def list_serial_candidates():
    ports = []
    if list_ports:
        ports = [p.device for p in list_ports.comports()]
    for p in ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']:
        if p not in ports:
            ports.append(p)
    return ports

# ==========================
# Traffic light helpers
# ==========================
def normalize_lighting(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    return out

def traffic_light_state(frame_bgr: np.ndarray, tl_roi):
    """
    Returns: (state, g_ratio, r_ratio, roi, g_mask, r_mask)
      state: "GREEN", "RED", "NONE"
    """
    (y0, y1, x0, x1) = tl_roi
    roi = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # GREEN HSV
    g_mask = cv2.inRange(hsv, (35, 80, 70), (90, 255, 255))

    # RED HSV (two hue ranges)
    r1 = cv2.inRange(hsv, (0, 90, 70), (10, 255, 255))
    r2 = cv2.inRange(hsv, (170, 90, 70), (179, 255, 255))
    r_mask = cv2.bitwise_or(r1, r2)

    g_count = int(cv2.countNonZero(g_mask))
    r_count = int(cv2.countNonZero(r_mask))

    area = int(roi.shape[0] * roi.shape[1])
    g_ratio = g_count / max(1, area)
    r_ratio = r_count / max(1, area)

    # Thresholds (tune if needed)
    if g_ratio > 0.015 and g_count > 150:
        return "GREEN", g_ratio, r_ratio, roi, g_mask, r_mask
    if r_ratio > 0.012 and r_count > 150:
        return "RED", g_ratio, r_ratio, roi, g_mask, r_mask

    return "NONE", g_ratio, r_ratio, roi, g_mask, r_mask

# ==========================
# Follow-the-gap helpers
# ==========================
def find_largest_gap(free_mask_1d, min_gap):
    best = None
    best_len = 0
    start = None
    for i, v in enumerate(free_mask_1d):
        if v and start is None:
            start = i
        if (not v or i == len(free_mask_1d) - 1) and start is not None:
            end = i if (v and i == len(free_mask_1d) - 1) else i - 1
            seg_len = end - start + 1
            if seg_len >= min_gap and seg_len > best_len:
                best_len = seg_len
                best = (start, end)
            start = None
    return best

def auto_canny(img_u8, sigma=0.33):
    v = float(np.median(img_u8))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_u8, lower, upper), lower, upper

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

# ==========================
# Track (orange wall + black line) helpers
# ==========================
def track_masks_from_roi(roi_bgr):
    """
    Returns:
      orange_mask, black_mask, track_mask
    - orange: HSV range (wide for shiny fabric)
    - black: adaptive threshold on grayscale (works on bright floor)
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # ORANGE wall (wide to tolerate reflections)
    o1 = cv2.inRange(hsv, (5,  40, 40), (28, 255, 255))
    o2 = cv2.inRange(hsv, (0,  35, 35), (35, 255, 255))
    orange = cv2.bitwise_or(o1, o2)

    # ✅ BLACK line on bright ground (adaptive)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    black = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    # Clean up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, k, iterations=2)
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN,  k, iterations=1)

    black  = cv2.morphologyEx(black,  cv2.MORPH_OPEN,  k, iterations=1)
    black  = cv2.morphologyEx(black,  cv2.MORPH_CLOSE, k, iterations=2)

    track = cv2.bitwise_or(orange, black)
    return orange, black, track

def black_line_x(black_mask):
    """
    Returns x centroid of black line if strong enough, else None
    """
    cnt = int(cv2.countNonZero(black_mask))
    if cnt < 250:
        return None
    M = cv2.moments(black_mask, binaryImage=True)
    if abs(M["m00"]) < 1e-6:
        return None
    cx = int(M["m10"] / M["m00"])
    return cx

def orange_borders_mid_x(orange_mask):
    """
    Estimate left/right orange borders and return midpoint x if possible.
    Uses a lower band (near car) for stability.
    """
    h, w = orange_mask.shape[:2]
    y1 = int(h * 0.55)
    y2 = int(h * 0.98)
    band = orange_mask[y1:y2, :]

    col = np.sum(band > 0, axis=0).astype(np.int32)
    if col.max() < 5:
        return None, None, None

    thresh = max(6, int(0.15 * (y2 - y1)))
    xs = np.where(col >= thresh)[0]
    if xs.size < 20:
        return None, None, None

    left = int(xs[0])
    right = int(xs[-1])
    if right - left < 40:
        return None, None, None

    mid = (left + right) // 2
    return left, right, mid

# ==========================
# Args
# ==========================
ap = argparse.ArgumentParser(description="Traffic Light gating + Camera Follow-the-Gap (single camera)")
ap.add_argument('--serial', default=None, help='Arduino serial port (e.g. /dev/ttyACM0). If not provided auto-detect')
ap.add_argument('--baud', type=int, default=115200, help='Serial baud rate')
ap.add_argument('--no-arduino', action='store_true', help="Run without Arduino (dry-run)")

# Speed / steering
ap.add_argument('--fast-speed', type=int, default=180)
ap.add_argument('--slow-speed', type=int, default=115)
ap.add_argument('--max-steer', type=int, default=100)
ap.add_argument('--steer-gain', type=float, default=1.35)
ap.add_argument('--steer-smooth', type=float, default=0.25)
ap.add_argument('--speed-smooth', type=float, default=0.20)

# Camera
ap.add_argument('--cam', type=int, default=0)
ap.add_argument('--width', type=int, default=640)
ap.add_argument('--height', type=int, default=480)
ap.add_argument('--fps', type=int, default=30)

# ROI (FTG)
ap.add_argument('--roi-top', type=float, default=0.55, help='FTG ROI top fraction (0..1). Try 0.45 for earlier objects.')

# Vision (EDGE-BASED)
ap.add_argument('--blur', type=int, default=5)
ap.add_argument('--use-auto-canny', action='store_true')
ap.add_argument('--canny1', type=int, default=60)
ap.add_argument('--canny2', type=int, default=150)
ap.add_argument('--auto-canny-sigma', type=float, default=0.33)
ap.add_argument('--use-gradient-edges', action='store_true')

ap.add_argument('--dilate', type=int, default=7)
ap.add_argument('--obstacle-thresh', type=int, default=18)
ap.add_argument('--min-gap', type=int, default=60)

ap.add_argument('--clahe', action='store_true')
ap.add_argument('--clahe-clip', type=float, default=2.0)
ap.add_argument('--clahe-grid', type=int, default=8)

ap.add_argument('--clearance-px', type=int, default=18)
ap.add_argument('--wall-follow', action='store_true')
ap.add_argument('--wall-bias', type=float, default=0.35)
ap.add_argument('--danger-thresh', type=float, default=70.0)

# Traffic light voting
ap.add_argument('--tl-window', type=int, default=10)
ap.add_argument('--tl-green-min', type=int, default=6)
ap.add_argument('--tl-red-min', type=int, default=5)

# Traffic light ROI (defaults full frame)
ap.add_argument('--tl-y0', type=float, default=0.0, help='0..1 (fraction of height)')
ap.add_argument('--tl-y1', type=float, default=1.0, help='0..1 (fraction of height)')
ap.add_argument('--tl-x0', type=float, default=0.0, help='0..1 (fraction of width)')
ap.add_argument('--tl-x1', type=float, default=1.0, help='0..1 (fraction of width)')

# Preview
ap.add_argument('--preview', action='store_true', help='Show debug windows')
args = ap.parse_args()

# ==========================
# Arduino serial
# ==========================
arduino = None
if not args.no_arduino:
    candidates = []
    if args.serial:
        candidates.append(args.serial)
    candidates.extend(list_serial_candidates())

    opened = False
    for p in candidates:
        try:
            arduino = serial.Serial(p, args.baud, timeout=0.05)
            time.sleep(2)
            print(f"✅ Opened Arduino serial port: {p}")
            opened = True
            break
        except Exception as e:
            print(f"Couldn't open {p}: {e}")

    if not opened:
        print("⚠️ Warning: No Arduino serial port available. Running in dry-run mode.")
        arduino = None
else:
    print("Running without Arduino (--no-arduino). Commands will not be sent.")

def send_cmd(steering, speed):
    steering = int(np.clip(steering, -args.max_steer, args.max_steer))
    speed = int(np.clip(speed, 0, 255))
    cmd = f"{steering},{speed}\n"
    if arduino and arduino.is_open:
        try:
            arduino.write(cmd.encode())
            try:
                arduino.flush()
            except Exception:
                pass
        except Exception as e:
            print(f"Error writing to Arduino: {e}")
    else:
        print(f"DRY: {cmd.strip()}")

# ==========================
# Camera init
# ==========================
cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS, args.fps)
try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
except Exception:
    pass

if not cap.isOpened():
    print("❌ Error: camera not available")
    send_cmd(0, 0)
    sys.exit(1)

FAST_SPEED = args.fast_speed
SLOW_SPEED = args.slow_speed

print("🚗 Combo started: Traffic Light gates Follow-the-Gap (GREEN LATCH MODE)")
print("   ✅ Initially STOP until GREEN is detected")
print("   ✅ After GREEN: keep running (do NOT go back to STOP)")
print("   ✅ Track aware: ignores ORANGE wall + follows BLACK line / center")
print("   Press 'q' to quit")

# TL vote smoothing
tl_hist = deque(maxlen=args.tl_window)

green_latched = False
run_enabled = False

steer_f = 0.0
speed_f = 0.0
last_target_x = None
last_print = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        H, W = frame.shape[:2]

        # ---------- Traffic Light ----------
        y0 = int(H * clamp01(args.tl_y0))
        y1 = int(H * clamp01(args.tl_y1))
        x0 = int(W * clamp01(args.tl_x0))
        x1 = int(W * clamp01(args.tl_x1))
        y1 = max(y1, y0 + 2)
        x1 = max(x1, x0 + 2)

        frame_n = normalize_lighting(frame)
        tl_state, g_ratio, r_ratio, tl_roi_img, g_mask, r_mask = traffic_light_state(
            frame_n, (y0, y1, x0, x1)
        )

        tl_hist.append(tl_state)
        green_votes = sum(1 for x in tl_hist if x == "GREEN")
        red_votes   = sum(1 for x in tl_hist if x == "RED")

        # GREEN latch
        if (not green_latched) and (green_votes >= args.tl_green_min):
            green_latched = True
        run_enabled = green_latched

        # ---------- STOP until GREEN latched ----------
        if not run_enabled:
            steer_f = (1.0 - args.steer_smooth) * 0.0 + args.steer_smooth * steer_f
            speed_f = (1.0 - args.speed_smooth) * 0.0 + args.speed_smooth * speed_f
            send_cmd(int(steer_f), int(speed_f))
            action = "STOP (waiting for GREEN latch)"

            if args.preview:
                dbg = frame_n.copy()
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (255, 255, 255), 2)
                cv2.putText(dbg, f"TL:{tl_state} LATCH:{green_latched} RUN:{run_enabled}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(dbg, f"G:{g_ratio:.3f} R:{r_ratio:.3f} votes G:{green_votes} R:{red_votes}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(dbg, action, (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Camera Preview (Combo)", dbg)
                cv2.imshow("Green Mask", g_mask)
                cv2.imshow("Red Mask", r_mask)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            else:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            now = time.time()
            if now - last_print > 0.25:
                print(f"TL={tl_state:4s} votes(G,R)=({green_votes},{red_votes}) "
                      f"ratios(G,R)=({g_ratio:.3f},{r_ratio:.3f}) green_latched={green_latched} -> STOP")
                last_print = now
            continue

        # ---------- Follow-the-Gap (RUN) ----------
        roi_y = int(H * np.clip(args.roi_top, 0.0, 0.95))
        roi = frame[roi_y:H, :]

        # Track masks
        orange_mask, black_mask, track_mask = track_masks_from_roi(roi)
        line_cx = black_line_x(black_mask)
        left_o, right_o, mid_o = orange_borders_mid_x(orange_mask)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if args.clahe:
            clahe = cv2.createCLAHE(
                clipLimit=float(args.clahe_clip),
                tileGridSize=(int(args.clahe_grid), int(args.clahe_grid))
            )
            gray = clahe.apply(gray)

        k = args.blur if args.blur % 2 == 1 else args.blur + 1
        k = max(3, k)
        gray_blur = cv2.GaussianBlur(gray, (k, k), 0)

        if args.use_auto_canny:
            edges, lo, hi = auto_canny(gray_blur, args.auto_canny_sigma)
        else:
            edges = cv2.Canny(gray_blur, args.canny1, args.canny2)
            lo, hi = args.canny1, args.canny2

        if args.use_gradient_edges:
            gx = cv2.Sobel(gray_blur, cv2.CV_16S, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_blur, cv2.CV_16S, 0, 1, ksize=3)
            mag = cv2.convertScaleAbs(cv2.abs(gx) + cv2.abs(gy))
            _, gbin = cv2.threshold(mag, 35, 255, cv2.THRESH_BINARY)
            edges = cv2.bitwise_or(edges, gbin)

        # ✅ Remove track from edges so orange wall is NOT an obstacle (strong removal)
        tk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        track_fat = cv2.dilate(track_mask, tk, iterations=3)
        edges_clean = cv2.bitwise_and(edges, cv2.bitwise_not(track_fat))

        d = args.dilate if args.dilate % 2 == 1 else args.dilate + 1
        d = max(3, d)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        obs = cv2.dilate(edges_clean, kernel, iterations=1)
        obs = cv2.morphologyEx(obs, cv2.MORPH_CLOSE, kernel, iterations=1)

        thick = cv2.dilate(obs, kernel, iterations=1)
        free_bin = cv2.threshold(thick, 1, 255, cv2.THRESH_BINARY_INV)[1]
        dist = cv2.distanceTransform(free_bin, cv2.DIST_L2, 5)

        col_density = np.mean(obs, axis=0)
        col_density = cv2.GaussianBlur(col_density.reshape(1, -1), (1, 31), 0).reshape(-1)
        free_cols = col_density < args.obstacle_thresh

        gap = find_largest_gap(free_cols, args.min_gap)

        steering = 0
        speed = FAST_SPEED
        action = "FORWARD"

        center_x = W // 2
        mid = col_density[W//3:2*W//3] if W >= 6 else col_density
        danger = float(np.max(mid)) if len(mid) else float(np.max(col_density))

        if gap is None:
            target_x = line_cx if line_cx is not None else (mid_o if mid_o is not None else center_x)
            err = (target_x - center_x) / max(1, center_x)
            steering = int(np.clip(err * args.steer_gain * args.max_steer, -args.max_steer, args.max_steer))
            speed = 0
            action = "NO GAP → STOP (track steer held)"
        else:
            gs, ge = gap
            gap_center = (gs + ge) // 2

            roi_h = roi.shape[0]
            y1b = int(roi_h * 0.45)
            y2b = int(roi_h * 0.95)
            band = dist[y1b:y2b, :]
            band_gap = band[:, gs:ge+1]

            col_clear = np.max(band_gap, axis=0)
            col_clear = cv2.GaussianBlur(col_clear.reshape(1, -1), (1, 21), 0).reshape(-1)
            xs = np.arange(gs, ge+1)

            if last_target_x is None:
                last_target_x = gap_center
            stab = -np.abs(xs - last_target_x) / max(1, W)

            prog = -0.15 * (np.abs(xs - center_x) / max(1, center_x))

            denom = np.max(col_clear)
            denom = denom if denom > 1e-6 else 1.0
            clear_term = col_clear / denom

            score = 1.0 * clear_term + 0.45 * stab + 0.25 * prog
            best_i = int(np.argmax(score))
            ftg_target = int(xs[best_i])

            # Track preference: BLACK line > ORANGE midpoint > none
            track_target = None
            if line_cx is not None:
                track_target = line_cx
            elif mid_o is not None:
                track_target = mid_o

            if track_target is not None:
                target_x = int(0.65 * track_target + 0.35 * ftg_target)
                target_x = int(np.clip(target_x, gs, ge))
            else:
                target_x = ftg_target

            if args.wall_follow:
                left_density = float(np.mean(col_density[:W//2]))
                right_density = float(np.mean(col_density[W//2:]))

                bias_dir = 0
                if left_density > right_density * 1.10:
                    bias_dir = +1
                elif right_density > left_density * 1.10:
                    bias_dir = -1

                diff = abs(left_density - right_density) / max(1.0, (left_density + right_density) / 2.0)
                bias_px = int(args.wall_bias * diff * (W * 0.10))
                target_x = int(np.clip(target_x + bias_dir * bias_px, gs, ge))

            sample_y = int(roi_h * 0.85)
            target_clear = float(dist[sample_y, target_x])

            if target_clear < args.clearance_px:
                if target_x < center_x:
                    target_x = min(ge, target_x + int(args.clearance_px))
                else:
                    target_x = max(gs, target_x - int(args.clearance_px))
                action = "CLEARANCE PUSH"

            err = (target_x - center_x) / max(1, center_x)
            steering = int(np.clip(err * args.steer_gain * args.max_steer, -args.max_steer, args.max_steer))

            if danger > args.danger_thresh or abs(steering) > (args.max_steer * 0.55) or target_clear < (args.clearance_px * 1.2):
                speed = SLOW_SPEED
                action = f"SLOW | danger={danger:.1f} clear={target_clear:.1f}"
            else:
                speed = FAST_SPEED
                action = f"FAST | danger={danger:.1f} clear={target_clear:.1f}"

            last_target_x = target_x

        steer_f = (1.0 - args.steer_smooth) * float(steering) + args.steer_smooth * steer_f
        speed_f = (1.0 - args.speed_smooth) * float(speed) + args.speed_smooth * speed_f

        send_cmd(int(steer_f), int(speed_f))

        now = time.time()
        if now - last_print > 0.25:
            print(
                f"TL={tl_state:4s} votes(G,R)=({green_votes},{red_votes}) "
                f"ratios(G,R)=({g_ratio:.3f},{r_ratio:.3f}) green_latched={green_latched} | "
                f"line_cx={line_cx} orange_mid={mid_o} | steer={int(steer_f)} speed={int(speed_f)} | {action}"
            )
            last_print = now

        if args.preview:
            dbg = frame_n.copy()
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (255, 255, 255), 2)
            cv2.line(dbg, (0, roi_y), (W, roi_y), (255, 255, 0), 2)

            if gap is not None:
                gs, ge = gap
                cv2.line(dbg, (gs, roi_y + 10), (ge, roi_y + 10), (0, 255, 0), 6)
                cx = W // 2
                cv2.line(dbg, (cx, roi_y + 10), (cx, roi_y + 40), (255, 0, 255), 2)
                if last_target_x is not None:
                    cv2.circle(dbg, (int(last_target_x), roi_y + 10), 8, (0, 255, 255), -1)

            if left_o is not None and right_o is not None:
                cv2.line(dbg, (left_o,  roi_y + 5), (left_o,  roi_y + 70), (0, 165, 255), 2)
                cv2.line(dbg, (right_o, roi_y + 5), (right_o, roi_y + 70), (0, 165, 255), 2)
            if line_cx is not None:
                cv2.circle(dbg, (line_cx, roi_y + 35), 7, (255, 255, 255), -1)

            cv2.putText(dbg, f"TL:{tl_state} LATCH:{green_latched} RUN:{run_enabled}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(dbg, f"G:{g_ratio:.3f} R:{r_ratio:.3f} votes G:{green_votes} R:{red_votes}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(dbg, f"{action}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(dbg, f"steer={int(steer_f)} speed={int(speed_f)} | canny={lo}-{hi}", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Camera Preview (Combo)", dbg)
            cv2.imshow("Obstacle Mask (ROI, cleaned)", obs)
            cv2.imshow("ROI Orange Mask", orange_mask)
            cv2.imshow("ROI Black Mask", black_mask)
            cv2.imshow("ROI Track Mask", track_mask)
            cv2.imshow("Green Mask", g_mask)
            cv2.imshow("Red Mask", r_mask)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    print("Stopping: sending 0,0 and cleaning up")
    try:
        send_cmd(0, 0)
    except Exception:
        pass
    if arduino and arduino.is_open:
        try:
            arduino.close()
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Stopped safely")
