"""
Casting Defect Detection - Image Upload Version
Upload gambar impeller dan deteksi cacat casting
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

class CastingDefectDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Casting Defect Detection - Upload Image")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        
        # Load model
        self.model = None
        self.current_image_path = None
        self.original_image = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2C3E50", height=80)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame,
            text="Submersible Pump Impeller Inspection System",
            font=("Arial", 24, "bold"),
            bg="#2C3E50",
            fg="white"
        )
        title_label.pack(pady=15)
        
       
        # Main container
        main_frame = tk.Frame(self.root, bg="#ECF0F1")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Image display frame
        image_frame = tk.LabelFrame(
            main_frame,
            text="📸 Impeller Image",
            font=("Arial", 14, "bold"),
            bg="#ECF0F1",
            fg="#2C3E50"
        )
        image_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(
            image_frame,
            text="No image loaded\n\nClick 'Upload Image' to start",
            font=("Arial", 14),
            bg="white",
            fg="#95A5A6",
            relief=tk.SUNKEN,
            width=40,
            height=20
        )
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg="#ECF0F1", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Upload button
        upload_btn = tk.Button(
            control_frame,
            text="📁 Upload Image",
            font=("Arial", 14, "bold"),
            bg="#3498DB",
            fg="white",
            activebackground="#2980B9",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=15,
            cursor="hand2",
            command=self.upload_image
        )
        upload_btn.pack(pady=(0, 10), fill=tk.X)
        
        # Predict button
        self.predict_btn = tk.Button(
            control_frame,
            text="🔍 Analyze Defect",
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
            state=tk.DISABLED,
            command=self.predict_image
        )
        self.predict_btn.pack(pady=(0, 20), fill=tk.X)
        
        # Result frame
        result_frame = tk.LabelFrame(
            control_frame,
            text="📊 Inspection Result",
            font=("Arial", 12, "bold"),
            bg="#ECF0F1",
            fg="#2C3E50"
        )
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Result display
        self.result_label = tk.Label(
            result_frame,
            text="Awaiting analysis...",
            font=("Arial", 16, "bold"),
            bg="white",
            fg="#95A5A6",
            relief=tk.SUNKEN,
            height=3
        )
        self.result_label.pack(padx=10, pady=10, fill=tk.X)
        
        # Confidence
        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: N/A",
            font=("Arial", 12),
            bg="#ECF0F1",
            fg="#34495E"
        )
        self.confidence_label.pack(pady=(0, 5))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            result_frame,
            mode='determinate',
            length=250
        )
        self.progress.pack(pady=10)
        
        # Details
        details_frame = tk.Frame(result_frame, bg="white")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.details_text = tk.Text(
            details_frame,
            font=("Courier", 9),
            bg="white",
            fg="#2C3E50",
            relief=tk.FLAT,
            height=10,
            wrap=tk.WORD
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.insert("1.0", "📋 Inspection Details:\n\n• Status: Waiting\n• Product: Impeller casting\n• View: Top view\n• Model: MobileNetV2")
        self.details_text.config(state=tk.DISABLED)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#34495E", height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(
            status_frame,
            text="Status: Ready | Model: Not Loaded",
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
            self.status_label.config(text="Status: Ready | Model: Loaded ✅")
            messagebox.showinfo("Success", "Model loaded successfully!\n\nReady for defect detection.")
        except Exception as e:
            self.status_label.config(text="Status: Error | Model: Not Loaded ❌")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}\n\nPlease ensure 'qc_inspector_model.h5' exists.")
    
    def upload_image(self):
        """Upload and display image"""
        file_path = filedialog.askopenfilename(
            title="Select Impeller Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                self.current_image_path = file_path
                img = Image.open(file_path)
                self.original_image = img.copy()
                
                # Resize for display
                display_size = (400, 400)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                # Enable predict button
                self.predict_btn.config(state=tk.NORMAL)
                
                # Update status
                self.status_label.config(text=f"Status: Image loaded | File: {os.path.basename(file_path)}")
                
                # Reset result
                self.result_label.config(text="Ready to analyze", bg="white", fg="#95A5A6")
                self.confidence_label.config(text="Confidence: N/A")
                self.progress['value'] = 0
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def predict_image(self):
        """Predict casting defect"""
        if not self.current_image_path or not self.model:
            messagebox.showwarning("Warning", "Please load an image and ensure model is loaded.")
            return
        
        try:
            # Update status
            self.status_label.config(text="Status: Analyzing...")
            self.predict_btn.config(state=tk.DISABLED)
            self.progress['value'] = 0
            self.root.update()
            
            # Preprocess image
            self.progress['value'] = 30
            self.root.update()
            
            img = image.load_img(self.current_image_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            self.progress['value'] = 60
            self.root.update()
            
            prediction = self.model.predict(img_array, verbose=0)
            score = prediction[0][0]
            
            self.progress['value'] = 90
            self.root.update()
            
            # Interpret result
            is_defective = score < 0.5
            result_text = "DEFECTIVE ❌" if is_defective else "OK ✅"
            confidence = (1 - score) * 100 if is_defective else score * 100
            
            # Update UI
            if is_defective:
                bg_color = "#E74C3C"  # Red
                fg_color = "white"
                status_text = "REJECT - Defective Casting"
            else:
                bg_color = "#27AE60"  # Green
                fg_color = "white"
                status_text = "PASS - Quality OK"
            
            self.result_label.config(text=result_text, bg=bg_color, fg=fg_color)
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            
            # Update details
            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete("1.0", tk.END)
            
            details = f"""📋 Inspection Report:

• Status: {status_text}
• Confidence: {confidence:.2f}%
• Prediction Score: {score:.4f}
• Threshold: 0.5

Product Information:
• Type: Submersible pump impeller
• View: Top view casting
• Model: MobileNetV2 (Transfer Learning)

"""
            
            if is_defective:
                details += """⚠️ Possible Defects:
• Blow holes (air pockets)
• Pinholes (small holes)
• Burr (unwanted protrusions)
• Shrinkage defects
• Surface defects
• Mould material defects

Recommendation:
❌ REJECT - Send for rework or scrap
"""
            else:
                details += """✅ Quality Check:
• No visible defects detected
• Surface quality: Good
• Shape integrity: Good
• Casting quality: Acceptable

Recommendation:
✅ PASS - Proceed to next stage
"""
            
            self.details_text.insert("1.0", details)
            self.details_text.config(state=tk.DISABLED)
            
            # Complete
            self.progress['value'] = 100
            self.predict_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Status: Analysis Complete | Result: {result_text}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.predict_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Error during analysis ❌")

def main():
    root = tk.Tk()
    app = CastingDefectDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
