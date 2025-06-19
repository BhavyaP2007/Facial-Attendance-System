import cv2
import torch
from torchvision import transforms
import time
import pyttsx3
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import model_facenet
from pathlib import Path
import pandas as pd
import threading
root_dir = Path(__file__).parent
photos = root_dir / "photos"
df = pd.read_csv(root_dir/"students.csv")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
def capture_from_box(n_seconds=10):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("❌ Error: Cannot access camera.")
        return
    print("📷 Camera started. Position your face in the box. Press 'q' to quit.")
    last_capture_time = time.time()
    box_w, box_h = 400, 500
    center_x, center_y =  1920// 2, 1080 // 2
    top_left = (center_x - box_w // 2, center_y - box_h // 2)
    bottom_right = (center_x + box_w // 2, center_y + box_h // 2)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break
        current_time = time.time()
        time_elapsed = current_time - last_capture_time
        time_remaining = max(0,int(n_seconds-time_elapsed))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, "Align face in box", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"⏳ Capturing in: {time_remaining}s", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("📸 Attendance Camera", frame)
        if current_time - last_capture_time >= n_seconds:
            print("📥 Capturing frame...")
            x1, y1 = top_left
            x2, y2 = bottom_right
            face_crop = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            def voice(name):
                t_1 = time.time()
                engine = pyttsx3.init()
                t_2 = time.time()
                print("Time for engine initialisation :",t_2-t_1)
                engine.say(f"{name} Present")
                engine.runAndWait()
            def attendance_mark(time,date,adm):
                flag = True
                f = open(root_dir / "attendance.csv","r",newline="")
                reader = csv.reader(f)
                reader = list(reader)[1:]
                for i in reader:
                    if len(i)==3:
                        if i[1] == date and int(i[2]) == int(adm):
                            flag = False
                if flag:        
                    f = open(root_dir / "attendance.csv","a",newline="")
                    writer = csv.writer(f)
                    writer.writerow([time,date,adm])
                    f.close()  
                    print("Actually marked")
            t1 = time.time()      
            # Convert to PyTorch tensor
            def recognize_face(tensor_frame):
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                tensor_frame = tensor_frame.to(device = device)
                tensor_frame = model_facenet.model(tensor_frame).squeeze()
                datetime_today = datetime.today().strftime(format="%d/%m/%Y %H:%M:%S")
                date_today = datetime.today().strftime(format="%d/%m/%Y")
                similarity = []
                for i in next(iter(os.walk(photos)))[2]:
                    title = i.split(".")[0]
                    img_path = photos / i
                    similarity.append((model_facenet.dot_product(img_path,tensor_frame).item(), title))
                similarity = sorted(similarity,key=lambda x: x[0],reverse=True)
                student = similarity[0]
                if student[0] > 0.5:
                    name_of_student = df.loc[df["adm_no"]==int(student[1])]["name"].item()
                    adm_of_student = df.loc[df["adm_no"]==int(student[1])]["adm_no"].item()
                    t2 = time.time()
                    print(f"Student {name_of_student}")
                    print("Time for data searching :",t2-t1)
                    threading.Thread(target=voice,args=(name_of_student,)).start()
                    threading.Thread(target=attendance_mark,
                                     args=(datetime_today,date_today,adm_of_student,)
                                     ).start()
                else:
                    print(similarity)   
            tensor_frame = preprocess(face_rgb).unsqueeze(0) 
            threading.Thread(target=recognize_face,args=(tensor_frame,)).start()
            last_capture_time = current_time + 2         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    capture_from_box(n_seconds=12)