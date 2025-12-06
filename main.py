import cv2
import torch
from collections import deque
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.traffic_light_cnn import TrafficLightCNN

VIDEO_PATH = "videos/video3.mp4"
OUTPUT_PATH = "output/result.mp4"

# --- Models ---
yolo_model = YOLO('weights/yolov8m.pt')

cnn_model = TrafficLightCNN()
cnn_model.load_state_dict(torch.load('weights/traffic_light_cnn.pth', map_location='cpu'))
cnn_model.eval()

cnn_classes = ['green', 'red', 'yellow']
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

device = torch.device('cpu')
cnn_model.to(device)

# --- Tracker ---
tracker = DeepSort(max_age=100, n_init=5)
prev_centroids = {}
violated_ids = set()

# --- Video capture ---
cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 30   # любое адекватное значение


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# --- Filter settings ---
MIN_LIGHT_AREA = 2000  # the minimum area of a traffic light to be considered a primary one

# majority vote — the queue of the last N colors
COLOR_HISTORY = deque(maxlen=5)

# ===================== LINE SELECTION =====================
ret, first_frame = cap.read()
if not ret:
    print("Error: Failed to open video.")
    exit()

points = []
temp_frame = first_frame.copy()

def draw_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)

cv2.namedWindow("Violation Detection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Violation Detection", draw_line)

while True:
    display_frame = temp_frame.copy()
    if len(points) == 2:
        cv2.line(display_frame, points[0], points[1], (0,0,255), 2)
    cv2.imshow("Violation Detection", display_frame)
    key = cv2.waitKey(1)
    if key == 13 and len(points) == 2:  # Enter
        break
    elif key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

line_start, line_end = points
line_y = (line_start[1] + line_end[1]) // 2
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("🚦 Video processing has begun...")


# ===================== Main loop =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    detections = []

    current_frame_colors = []

    # --- Detection ---
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()

        for box, cls_id in zip(boxes, cls_ids):
            x1, y1, x2, y2 = map(int, box)

            # ===========================
            #     TRAFFIC LIGHT FILTER
            # ===========================
            if cls_id == 9:  # Traffic light
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_LIGHT_AREA:
                    continue  # ignore small side traffic lights

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    roi_t = transform(roi_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = cnn_model(roi_t)
                        _, pred = torch.max(output, 1)
                        color = cnn_classes[pred.item()]

                    current_frame_colors.append(color)

            # ===========================
            #     TRANSPORT
            # ===========================
            elif cls_id in [2,3,5,7]:
                detections.append(([x1, y1, x2-x1, y2-y1], 0.9, cls_id))

    # ===========================
    # MAJORITY VOTE
    # ===========================
    if current_frame_colors:
        # We take the most common color among large traffic lights
        from statistics import mode
        try:
            frame_color = mode(current_frame_colors)
        except:
            frame_color = current_frame_colors[0]

        COLOR_HISTORY.append(frame_color)

    # If there is a history, we choose the most common color
    if len(COLOR_HISTORY) > 0:
        from statistics import mode
        try:
            light_color = mode(COLOR_HISTORY)
        except:
            light_color = COLOR_HISTORY[-1]
    else:
        light_color = None

    # ===========================
    #        TRACKING
    # ===========================
    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue

        track_id = t.track_id
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        if track_id in prev_centroids:
            prev_cy = prev_centroids[track_id][1]
            if prev_cy > line_y and cy <= line_y:
                if light_color == 'red':
                    violated_ids.add(track_id)
                    print(f"⚠️ Violation ID {track_id}")

        prev_centroids[track_id] = (cx, cy)

    # ===========================
    #       VISUALIZATION
    # ===========================
    display_frame = frame.copy()

    for t in tracks:
        if not t.is_confirmed():
            continue

        track_id = t.track_id
        x1, y1, x2, y2 = map(int, t.to_ltrb())

        if track_id in violated_ids:
            color = (0, 0, 255)
            label = "VIOLATION"
        else:
            color = (255, 255, 255)
            label = "Vehicle"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(display_frame, f"{label} ID:{track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    # traffic light status text
    if light_color:
        color_map = {'red': (0,0,255), 'green': (0,255,0), 'yellow': (0,255,255)}
        cv2.putText(display_frame, f"Light: {light_color.upper()}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color_map[light_color], 2)

    # line of violation
    cv2.line(display_frame, line_start, line_end, (0, 0, 255), 2)

    out.write(display_frame)

    cv2.imshow("Violation Detection", display_frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Processing complete. Violators: {len(violated_ids)}")
print(violated_ids)
