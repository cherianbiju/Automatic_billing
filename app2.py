import cv2
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import datetime
import tempfile
import img2pdf
import time
import winsound

load_dotenv()

model = YOLO(os.getenv('YOLO_MODEL_PATH'))
cap = cv2.VideoCapture(0)

detected_items = {}
last_detected_time = {}
item_positions = {}

supermarket_items = {
    'bottle': 70.00,
    'book': 40.00,
    'keyboard': 650.00,
    'cell phone': 35000.00,
    'remote': 250.00,
    'eggs': 180.00,
    'chocolate': 150.00,
    'tomato': 40.00,
    'potato': 20.00,
}

save_button_coords = None
quit_button_coords = None
exit_program = False

def save_invoice_as_pdf(sidebar_image):
    temp_image_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(temp_image_path, sidebar_image)
    pdf_path = f"Invoice_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(temp_image_path))
    os.remove(temp_image_path)
    print(f"Invoice saved as {pdf_path}")

def mouse_event(event, x, y, flags, param):
    global save_button_coords, quit_button_coords, exit_program, sidebar, detected_items
    frame_width = param['frame_width']
    x_sidebar = x - frame_width

    if event == cv2.EVENT_LBUTTONDOWN:
        if x_sidebar >= 0:
            if save_button_coords[0][0] <= x_sidebar <= save_button_coords[1][0] and save_button_coords[0][1] <= y <= save_button_coords[1][1]:
                save_invoice_as_pdf(sidebar)
            elif quit_button_coords[0][0] <= x_sidebar <= quit_button_coords[1][0] and quit_button_coords[0][1] <= y <= quit_button_coords[1][1]:
                exit_program = True
            else:
                for item, (pos_y, pos_x) in item_positions.items():
                    if pos_y - 15 <= y <= pos_y + 15 and 10 <= x_sidebar <= 290:
                        detected_items[item] -= 1
                        if detected_items[item] <= 0:
                            del detected_items[item]
                        break

cv2.namedWindow('Automatic Billing System')
ret, frame = cap.read()
if not ret:
    print("Failed to read from webcam.")
    cap.release()
    cv2.destroyAllWindows()
    exit()
frame_width = frame.shape[1]
cv2.setMouseCallback('Automatic Billing System', mouse_event, param={'frame_width': frame_width})

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = box.conf[0]

            if confidence > 0.5 and label in supermarket_items:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 50), 2)

                current_time = time.time()
                if label not in last_detected_time or (current_time - last_detected_time[label] > 2):
                    last_detected_time[label] = current_time
                    detected_items[label] = detected_items.get(label, 0) + 1
                    winsound.Beep(1000, 200)

    sidebar_width = 300
    sidebar_height = frame.shape[0]
    sidebar = np.ones((sidebar_height, sidebar_width, 3), dtype=np.uint8) * 255
    cv2.rectangle(sidebar, (0, 0), (sidebar_width - 1, sidebar_height - 1), (0, 0, 0), 2)

    y_offset = 40
    total_cost = 0
    item_positions.clear()

    cv2.putText(sidebar, 'INVOICE', (80, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 128), 2)
    y_offset += 40
    cv2.line(sidebar, (10, y_offset), (sidebar_width - 10, y_offset), (0, 0, 0), 2)
    y_offset += 30

    for item, count in detected_items.items():
        price = supermarket_items[item] * count
        text = f'{item.capitalize()} x{count}: Rs.{price:.2f}'
        cv2.putText(sidebar, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        item_positions[item] = (y_offset, 10)
        y_offset += 30
        total_cost += price

    y_offset += 10
    cv2.line(sidebar, (10, y_offset), (sidebar_width - 10, y_offset), (0, 0, 0), 2)
    y_offset += 40
    cv2.putText(sidebar, f'Total: Rs.{total_cost:.2f}', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    y_offset += 60

    save_button_coords = ((50, y_offset), (250, y_offset + 40))
    cv2.rectangle(sidebar, save_button_coords[0], save_button_coords[1], (0, 128, 0), -1)
    cv2.putText(sidebar, 'Save Invoice', (70, y_offset + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 60

    quit_button_coords = ((50, y_offset), (250, y_offset + 40))
    cv2.rectangle(sidebar, quit_button_coords[0], quit_button_coords[1], (0, 0, 128), -1)
    cv2.putText(sidebar, 'Quit', (110, y_offset + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined_frame = np.hstack((frame, sidebar))
    cv2.imshow('Automatic Billing System', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or exit_program:
        break

cap.release()
cv2.destroyAllWindows()
