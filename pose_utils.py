import math
import mediapipe as mp


def visible(landmarks, required):
    return all(landmarks[pt.value].visibility > 0.7 for pt in required)


def detect_side(landmarks, required):
    ls, rs, lh, rh = [landmarks[pt.value] for pt in required]
    x_vals = [ls.x, rs.x, lh.x, rh.x]
    if max(x_vals) - min(x_vals) < 0.03:
        return "Right" if (ls.z + lh.z) / 2 > (rs.z + rh.z) / 2 else "Left"
    return None


def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba == 0 or mag_bc == 0:
        return 0
    cos_angle = dot / (mag_ba * mag_bc)
    angle = math.acos(max(min(cos_angle, 1), -1))
    return math.degrees(angle)
