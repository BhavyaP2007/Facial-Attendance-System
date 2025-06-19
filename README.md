# 🧠 Face Recognition Attendance System

A real-time, AI-powered attendance system using webcam input. It recognizes student faces using a FaceNet-based deep learning model in PyTorch and records their attendance in a CSV file with voice announcements.

Features:
- Real-time face detection from webcam
- Face recognition using FaceNet embeddings
- Attendance logged with timestamp in attendance.csv
- Voice announcement using pyttsx3
- Automatically skips already-present students
- Lightweight and responsive UI

Folder Structure:
- photos/ — Passport-sized student photos (filename = admno.jpg)
- students.csv — Format: admno,name
- attendance.csv — Auto-generated attendance log
- model_facenet.py — FaceNet model & similarity function
- frame_capture.py — Main webcam & attendance script
- README.md

Requirements:
Install required Python libraries:
torch, torchvision, opencv-python, pandas, pyttsx3, matplotlib, pathlib,os,csv,threading,pillow,facenet_pytorch

How to Use:
1. Add Student Photos  
   Place passport-sized .jpg images in the photos/ folder.  
   Filename must be the Admission Number (e.g., 12345.jpg)

2. Add Student Records  
   In students.csv, add student data in the format:  
   admno,name  
   12345,Bhavya  
   67890,Aditi  

3. Run the Program  
   Run frame_capture.py to start the attendance system.  
   The webcam will open and show a green box — align the face inside it.  
   Every 10 seconds, a frame is captured and matched.  
   If recognized and not already marked, attendance is recorded and name is announced.

Output: attendance.csv  
Example entries:  
Timestamp, Date, Adm No  
16/06/2025 09:34:22, 16/06/2025, 12345  
16/06/2025 09:35:10, 16/06/2025, 67890

Notes:
- Ensure good lighting with a bright background and clear front-facing photos
- Use consistent naming for photos (only admno)
- You can adjust frame capture interval with n_seconds in the code

Contact:
Reach out if you wish to deploy this to schools, colleges, or offices.
