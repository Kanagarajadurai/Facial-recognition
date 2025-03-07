import cv2
import tkinter as tk
from tkinter import Label, Button, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.trainedfacemodel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Change to 0 to use the default webcam
        self.video_source = 0  # 0 refers to the default camera
        self.video_capture = cv2.VideoCapture(self.video_source)
        
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open video source.")
            self.root.quit()
        
        self.is_recording = False
        self.video_writer = None
        self.face_counter = 0
        self.face_collection_folder = "face_collection"
        
        if not os.path.exists(self.face_collection_folder):
            os.makedirs(self.face_collection_folder)

        self.video_label = Label(root)
        self.video_label.pack()
        
        self.start_button = Button(root, text="Start", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = Button(root, text="Stop", command=self.stop_stream)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.save_photo_button = Button(root, text="Save Photo", command=self.save_photo)
        self.save_photo_button.pack(side=tk.LEFT, padx=10)

        self.start_video_button = Button(root, text="Start Video", command=self.start_recording)
        self.start_video_button.pack(side=tk.LEFT, padx=10)

        self.stop_video_button = Button(root, text="Stop Video", command=self.stop_recording)
        self.stop_video_button.pack(side=tk.LEFT, padx=10)

        self.save_faces_button = Button(root, text="Save Face", command=self.save_face)
        self.save_faces_button.pack(side=tk.LEFT, padx=10)

        self.root.bind('<q>', self.stop_stream_key)
        
        self.update_frame()

    def update_frame(self):
        if self.video_capture.isOpened():
            ret, frame = self.video_capture.read()

            if not ret:
                self.video_capture.release()
                return

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.trainedfacemodel.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.config(image=photo)
            self.video_label.image = photo

            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)

            self.root.after(10, self.update_frame)

    def start_stream(self):
        if not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(self.video_source)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open video source.")
                return
        self.update_frame()

    def stop_stream(self):
        self.video_capture.release()
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        self.video_writer = None

    def stop_stream_key(self, event):
        self.stop_stream()

    def save_photo(self):
        if self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                filename = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
                if filename:
                    cv2.imwrite(filename, frame)

    def start_recording(self):
        if self.video_capture.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            filename = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("All files", "*.*")])
            if filename:
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (int(self.video_capture.get(3)), int(self.video_capture.get(4))))
                self.is_recording = True

    def stop_recording(self):
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.is_recording = False

    def save_face(self):
        if self.video_capture.isOpened():
            ret, frame = self.video_capture.read()

            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.trainedfacemodel.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face_img = frame[y:y + h, x:x + w]
                    self.face_counter += 1
                    filename = os.path.join(self.face_collection_folder, f"face_{self.face_counter}.jpg")
                    cv2.imwrite(filename, face_img)
                    messagebox.showinfo("Info", f"Face saved as {filename}")

    def __del__(self):
        self.video_capture.release()
        if self.is_recording and self.video_writer:
            self.video_writer.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()