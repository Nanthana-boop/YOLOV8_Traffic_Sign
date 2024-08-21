import cv2
import torch
import os  # สำหรับการสร้างโฟลเดอร์
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from datetime import datetime  # สำหรับการใส่เวลาลงในภาพ

def main():
    model = YOLO("final.pt")  # โหลดโมเดล YOLO
    cap = cv2.VideoCapture(0)

    # สร้างโฟลเดอร์ 'captures' หากยังไม่มี
    if not os.path.exists("captures"):
        os.makedirs("captures")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ลดขนาดของเฟรมเพื่อลดการใช้ทรัพยากร
        frame_resized = cv2.resize(frame, (640, 480))

        # ประมวลผลเฟรมที่ถูกย่อขนาดแล้วด้วยโมเดล YOLO
        results = model(frame_resized, conf=0.9)

        # สร้าง Annotator เพื่อใส่กรอบและป้ายชื่อ
        annotator = Annotator(frame_resized)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                # บันทึกภาพเมื่อพบวัตถุ
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # วาดกรอบและใส่ข้อมูลวัตถุและเวลาในกรอบ
                for box in boxes:
                    b = box.xyxy[0].tolist()  # แปลงเป็นลิสต์
                    c = int(box.cls)  # แปลง class ID เป็น integer
                    label = f"{model.names[c]} {timestamp}"
                    if box.conf >= 0.8:  # วาดกรอบเฉพาะที่มีความเชื่อมั่น >= 0.8
                        annotator.box_label(b, label)  # ใส่กรอบและป้ายชื่อ

                # บันทึกภาพที่มีการใส่กรอบและข้อความเรียบร้อยแล้ว
                output_frame = annotator.result()
                cv2.imwrite(f"captures/object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", output_frame)

        # แสดงผลภาพที่ถูกใส่กรอบแล้ว
        frame = annotator.result()

        # ปรับขนาดกลับไปเป็นขนาดเดิม
        frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))

        cv2.imshow("result", frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

