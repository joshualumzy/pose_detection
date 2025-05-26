import cv2
import mediapipe as mp
import pygame

from audio_utils import get_audio, clean_up_audio
from pose_utils import visible, detect_side, calculate_angle

pygame.mixer.init()
channel = pygame.mixer.Channel(0)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
REQUIRED = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
]

cap = cv2.VideoCapture(1)
last_spoken = None
side_locked = None
side_frame_counter = 0
reset_counter = 0
RESET_THRESHOLD = 50

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not (
            results.pose_landmarks
            and visible(results.pose_landmarks.landmark, REQUIRED)
        ):
            side_locked = None
            side_frame_counter = 0
            reset_counter = 0
            cv2.putText(
                image,
                "Move back: shoulders and hips must be visible",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4,
            )
        else:
            lm = results.pose_landmarks.landmark
            side = detect_side(lm, REQUIRED)

            if side in ["Left", "Right"]:
                if side_locked != side:
                    side_frame_counter += 1
                    if side_frame_counter > 10:
                        side_locked = side
                        reset_counter = 0
                        if last_spoken != side_locked:
                            channel.stop()
                            sound, audio_file = get_audio(side_locked)
                            channel.play(sound)
                            last_spoken = side_locked
                            clean_up_audio(audio_file)
                else:
                    reset_counter = 0
            else:
                side_frame_counter = 0
                reset_counter += 1
                if reset_counter > RESET_THRESHOLD:
                    side_locked = None
                    last_spoken = None

            if not side_locked:
                cv2.putText(
                    image,
                    "Please turn to the side",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    4,
                )

            view_color = (0, 255, 0) if side_locked else (0, 255, 255)
            view_text = f"View: {side_locked or 'Detecting...'}"
            cv2.putText(
                image, view_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, view_color, 4
            )

            angle = None
            if side_locked == "Left":
                elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            elif side_locked == "Right":
                elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

            if side_locked in ["Left", "Right"]:
                angle = calculate_angle(
                    (elbow.x, elbow.y), (shoulder.x, shoulder.y), (hip.x, hip.y)
                )

            if angle:
                cv2.putText(
                    image,
                    f"Shoulder Angle: {int(angle)} deg",
                    (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 0),
                    4,
                )

            diag_names = [
                "L_SHOULDER",
                "R_SHOULDER",
                "L_HIP",
                "R_HIP",
                "L_EAR",
                "R_EAR",
                "L_ELBOW",
                "R_ELBOW",
            ]
            diag_ids = [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_HIP.value,
                mp_pose.PoseLandmark.LEFT_EAR.value,
                mp_pose.PoseLandmark.RIGHT_EAR.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            ]
            for i, (idx, name) in enumerate(zip(diag_ids, diag_names)):
                lm_i = results.pose_landmarks.landmark[idx]
                info = f"{name}: x={lm_i.x:.2f}, y={lm_i.y:.2f}, z={lm_i.z:.2f}, v={lm_i.visibility:.2f}"
                y_pos = 270 + 60 * i
                cv2.putText(
                    image,
                    info,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    4,
                )

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Pose Debug", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
pygame.mixer.quit()
