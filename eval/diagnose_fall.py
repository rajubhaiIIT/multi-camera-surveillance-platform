import cv2
from ai.pose.detector import PoseDetector
from ai.pose.fall import FallDetector

det = PoseDetector('yolo11n-pose.pt', conf_thresh=0.25)

for seq, kind in [('fall-03-cam0.mp4', 'fall'), ('adl-02-cam0.mp4', 'adl'), ('adl-04-cam0.mp4', 'adl')]:
    path = f'data/raw/urfall/{seq}'
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fall_det = FallDetector()
    fidx = 0
    max_v = 0.0
    max_ang = 0.0
    max_asp = 0.0
    alerts = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ts = fidx / fps
        poses = det.detect(frame, camera_id='cam0', frame_idx=fidx)
        for p in poses:
            max_ang = max(max_ang, p.keypoints.torso_angle)
            max_asp = max(max_asp, p.keypoints.aspect_ratio)
            alert = fall_det.update(p, ts)
            if alert and alert.state.value == 'FALLEN':
                alerts.append((fidx, ts, alert.descent_velocity, alert.torso_angle, alert.aspect_ratio))
        fidx += 1
    cap.release()
    print(f'{seq} ({kind}): MaxAng={max_ang:.1f}, MaxAsp={max_asp:.2f}, Alerts={len(alerts)}')
    if alerts:
        print(f'   First alert at frame {alerts[0][0]}, ts={alerts[0][1]:.2f}s, vel={alerts[0][2]:.2f}, ang={alerts[0][3]:.1f}, asp={alerts[0][4]:.2f}')
