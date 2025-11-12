"""
Casting Defect Detection - Real-time Camera Version
Deteksi real-time menggunakan webcam/camera
"""

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import threading
import os

class RealtimeCastingDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Casting Defect Detection - Real-time Camera")
        self.root.geometry("1100x750")
        self.root.resizable(False, False)
        
        # Variables
        self.model = None
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.prediction_score = 0.5
        self.confidence = 0.0
        self.fps = 0
        self.frame_count = 0
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2C3E50", height=80)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame,
            text="📹 Real-time Casting Defect Detection",
            font=("Arial", 24, "bold"),
            bg="#2C3E50",
            fg="white"
        )
        title_label.pack(pady=15)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Live Inspection via Camera • Submersible Pump Impeller",
            font=("Arial", 12),
            bg="#2C3E50",
            fg="#BDC3C7"
        )
        subtitle_label.place(relx=0.5, rely=0.75, anchor="center")
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#ECF0F1")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Camera display frame
        camera_frame = tk.LabelFrame(
            main_frame,
            text="📹 Live Camera Feed",
            font=("Arial", 14, "bold"),
            bg="#ECF0F1",
            fg="#2C3E50"
        )
        camera_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(
            camera_frame,
            text="Camera not started\n\nClick 'Start Camera' to begin",
            font=("Arial", 14),
            bg="black",
            fg="white",
            relief=tk.SUNKEN
        )
        self.camera_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg="#ECF0F1", width=350)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Start/Stop button
        self.toggle_btn = tk.Button(
            control_frame,
            text="▶️ Start Camera",
            font=("Arial", 14, "bold"),
            bg="#27AE60",
            fg="white",
            activebackground="#229954",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=15,
            cursor="hand2",
            command=self.toggle_camera
        )
        self.toggle_btn.pack(pady=(0, 10), fill=tk.X)
        
        # Capture button
        self.capture_btn = tk.Button(
            control_frame,
            text="📸 Capture Frame",
            font=("Arial", 12, "bold"),
            bg="#3498DB",
            fg="white",
            activebackground="#2980B9",
            activeforeground="white",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.capture_frame
        )
        self.capture_btn.pack(pady=(0, 20), fill=tk.X)
        
        # Detection result frame
        result_frame = tk.LabelFrame(
            control_frame,
            text="🎯 Detection Result",
            font=("Arial", 12, "bold"),
            bg="#ECF0F1",
            fg="#2C3E50"
        )
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Result display
        self.result_label = tk.Label(
            result_frame,
            text="Waiting...",
            font=("Arial", 18, "bold"),
            bg="white",
            fg="#95A5A6",
            relief=tk.SUNKEN,
            height=2
        )
        self.result_label.pack(padx=10, pady=10, fill=tk.X)
        
        # Confidence bar
        confidence_container = tk.Frame(result_frame, bg="#ECF0F1")
        confidence_container.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            confidence_container,
            text="Confidence:",
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            fg="#34495E"
        ).pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(
            confidence_container,
            text="0.00%",
            font=("Arial", 10, "bold"),
            bg="#ECF0F1",
            fg="#27AE60"
        )
        self.confidence_label.pack(side=tk.RIGHT)
        
        self.confidence_bar = ttk.Progressbar(
            result_frame,
            mode='determinate',
            length=300
        )
        self.confidence_bar.pack(pady=5)
        
        # Statistics
        stats_frame = tk.LabelFrame(
            control_frame,
            text="📊 Statistics",
            font=("Arial", 11, "bold"),
            bg="#ECF0F1",
            fg="#2C3E50"
        )
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        stats_bg = tk.Frame(stats_frame, bg="white")
        stats_bg.pack(fill=tk.BOTH, padx=10, pady=10)
        
        self.fps_label = tk.Label(
            stats_bg,
            text="FPS: 0",
            font=("Arial", 10),
            bg="white",
            fg="#34495E",
            anchor=tk.W
        )
        self.fps_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.frames_label = tk.Label(
            stats_bg,
            text="Frames: 0",
            font=("Arial", 10),
            bg="white",
            fg="#34495E",
            anchor=tk.W
        )
        self.frames_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.camera_status_label = tk.Label(
            stats_bg,
            text="Camera: Inactive",
            font=("Arial", 10),
            bg="white",
            fg="#E74C3C",
            anchor=tk.W
        )
        self.camera_status_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.model_status_label = tk.Label(
            stats_bg,
            text="Model: Not Loaded",
            font=("Arial", 10),
            bg="white",
            fg="#E74C3C",
            anchor=tk.W
        )
        self.model_status_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#34495E", height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(
            status_frame,
            text="Status: Ready | Camera: Inactive | Model: Loading...",
            font=("Arial", 9),
            bg="#34495E",
            fg="white",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
    
    def load_model(self):
        """Load trained model"""
        try:
            model_path = '../qc_inspector_model.h5'
            if not os.path.exists(model_path):
                model_path = 'qc_inspector_model.h5'
            
            self.status_label.config(text="Status: Loading model...")
            self.root.update()
            
            self.model = tf.keras.models.load_model(model_path)
            self.model_status_label.config(text="Model: Loaded ✅", fg="#27AE60")
            self.status_label.config(text="Status: Ready | Model: Loaded ✅")
            messagebox.showinfo("Success", "Model loaded successfully!\n\nReady for real-time detection.")
        except Exception as e:
            self.model_status_label.config(text="Model: Error ❌", fg="#E74C3C")
            self.status_label.config(text="Status: Error | Model: Not Loaded ❌")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}\n\nPlease ensure 'qc_inspector_model.h5' exists.")
    
    def toggle_camera(self):
        """Start or stop camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera feed"""
        if not self.model:
            messagebox.showwarning("Warning", "Model not loaded. Please restart the application.")
            return
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not access camera.\nPlease check camera connection.")
                return
            
            self.is_running = True
            self.toggle_btn.config(text="⏸️ Stop Camera", bg="#E74C3C")
            self.capture_btn.config(state=tk.NORMAL)
            self.camera_status_label.config(text="Camera: Active ✅", fg="#27AE60")
            self.status_label.config(text="Status: Camera Running | Detecting...")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.process_camera, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")
    
    def stop_camera(self):
        """Stop camera feed"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.toggle_btn.config(text="▶️ Start Camera", bg="#27AE60")
        self.capture_btn.config(state=tk.DISABLED)
        self.camera_status_label.config(text="Camera: Inactive", fg="#E74C3C")
        self.status_label.config(text="Status: Camera Stopped")
        
        # Reset display
        self.camera_label.config(
            image="",
            text="Camera stopped\n\nClick 'Start Camera' to resume",
            bg="black",
            fg="white"
        )
    
    def process_camera(self):
        """Process camera frames"""
        import time
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Flip frame horizontally (mirror)
            frame = cv2.flip(frame, 1)
            
            # Store current frame
            self.current_frame = frame.copy()
            
            # Preprocess for prediction
            resized = cv2.resize(frame, (224, 224))
            img_array = resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = self.model.predict(img_array, verbose=0)
            self.prediction_score = prediction[0][0]
            
            # Determine result
            is_defective = self.prediction_score < 0.5
            self.confidence = (1 - self.prediction_score) * 100 if is_defective else self.prediction_score * 100
            
            # Draw overlay
            if is_defective:
                color = (0, 0, 255)  # Red (BGR)
                label = "DEFECTIVE"
                status = "REJECT"
            else:
                color = (0, 255, 0)  # Green (BGR)
                label = "OK"
                status = "PASS"
            
            # Draw rectangle and text
            cv2.rectangle(frame, (10, 10), (300, 120), color, 2)
            cv2.rectangle(frame, (10, 10), (300, 50), color, -1)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, f"{label} ({self.confidence:.1f}%)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Score: {self.prediction_score:.4f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Convert to RGB for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Update UI in main thread
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo
            
            # Update result
            if is_defective:
                self.result_label.config(text="DEFECTIVE ❌", bg="#E74C3C", fg="white")
            else:
                self.result_label.config(text="OK ✅", bg="#27AE60", fg="white")
            
            self.confidence_label.config(text=f"{self.confidence:.2f}%")
            self.confidence_bar['value'] = self.confidence
            
            # Update FPS
            fps_counter += 1
            self.frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            self.fps_label.config(text=f"FPS: {self.fps}")
            self.frames_label.config(text=f"Frames: {self.frame_count}")
            
            self.root.update()
    
    def capture_frame(self):
        """Capture current frame"""
        if self.current_frame is not None:
            try:
                # Create captures folder
                os.makedirs('captures', exist_ok=True)
                
                # Generate filename
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captures/capture_{timestamp}.jpg"
                
                # Save frame
                cv2.imwrite(filename, self.current_frame)
                
                messagebox.showinfo(
                    "Success",
                    f"Frame captured successfully!\n\nSaved to: {filename}\n\nResult: {'DEFECTIVE' if self.prediction_score < 0.5 else 'OK'}\nConfidence: {self.confidence:.2f}%"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to capture frame:\n{str(e)}")
    
    def on_closing(self):
        """Handle window close"""
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = RealtimeCastingDetector(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
