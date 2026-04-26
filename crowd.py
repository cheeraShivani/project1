import cv2
import pandas as pd
from ultralytics import YOLO
import pyttsx3
from datetime import datetime
import os

print("Crowd Monitoring System Started...")

# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- VOICE ----------------
engine = pyttsx3.init()
alert_given = False

# ---------------- EXCEL PATH (FIXED) ----------------
current_folder = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(current_folder, "crowd_log.xlsx")

print("Excel will be saved at:", excel_file)

# Create Excel file if not exists
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=["Date","Time","People_Count","Density","Status"])
    df.to_excel(excel_file, index=False)
    print("Excel file created")

# ---------------- MAIN LOOP ----------------
while True:

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    results = model(frame)

    person_count = 0

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.5:

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                person_count += 1

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,"Person",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # ---------------- DENSITY ----------------
    if person_count <= 3:
        density = "LOW"
        color = (0,255,0)
        status = "Safe"

    elif person_count <= 6:
        density = "MEDIUM"
        color = (0,255,255)
        status = "Crowd Building"

    else:
        density = "HIGH"
        color = (0,0,255)
        status = "High Crowd"

        if not alert_given:
            engine.say("Warning High Crowd Detected")
            engine.runAndWait()
            alert_given = True

    # ---------------- ARROW ----------------
    h, w = frame.shape[:2]

    if density == "HIGH":
        start = (w//2, h-50)
        end = (w//2, h//2)

        cv2.arrowedLine(frame, start, end, (0,0,255), 6, tipLength=0.3)

        cv2.putText(frame, "HIGH CROWD AREA",
                    (end[0]-200,end[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 3)

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, f"People Count: {person_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,255,255),3)

    cv2.putText(frame, f"Density: {density}",
                (20,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,color,3)

    cv2.imshow("Crowd Monitoring System", frame)

    # ---------------- SAVE TO EXCEL ----------------
    now = datetime.now()

    new_row = {
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "People_Count": person_count,
        "Density": density,
        "Status": status
    }

    try:
        df = pd.read_excel(excel_file)
    except:
        df = pd.DataFrame(columns=["Date","Time","People_Count","Density","Status"])

    df.loc[len(df)] = new_row
    df.to_excel(excel_file, index=False)

    print("Saved to Excel")

    # EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Program Ended")