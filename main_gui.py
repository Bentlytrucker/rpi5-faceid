
#!/usr/bin/env python3
"""
Raspberry Pi Face Recognition GUI - External Camera Window
Shows camera feed in separate OpenCV window, GUI for controls and info.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import threading
import time
import os
import sys

# Add parent directory to path to import inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import FaceRecognitionInference

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title()
        self.root.geometry("600x500")
        self.root.configure(bg='#2c3e50')
        
        # Initialize face recognition system
        self.inference = None
        self.is_running = False
        self.camera_thread = None
        self.current_frame = None
        self.recognition_completed = False
        self.recognition_start_time = None
        self.recognition_duration = 3.0  # 3 seconds required
        self.current_user = None
        
        # GUI variables
        self.status_label = None
        self.recognition_label = None
        self.registered_faces_label = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        

        
        # Control panel frame
        control_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop button
        self.start_button = tk.Button(
            control_frame,
            text="Login",
            command=self.toggle_recognition,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10
        )
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Register face button
        self.register_button = tk.Button(
            control_frame,
            text="Register Face (R)",
            command=self.register_face,
            font=('Arial', 12),
            bg='#3498db',
            fg='white',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.register_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Status: Ready",
            font=('Arial', 10),
            bg='#34495e',
            fg='white'
        )
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Information display frame
        info_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_title = tk.Label(
            info_frame,
            text="System Information",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        )
        info_title.pack(pady=10)
        
        # Recognition result
        recognition_frame = tk.Frame(info_frame, bg='#34495e')
        recognition_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            recognition_frame,
            text="Current Recognition:",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(anchor=tk.W)
        
        self.recognition_label = tk.Label(
            recognition_frame,
            text="No face detected",
            font=('Arial', 10),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.recognition_label.pack(anchor=tk.W, pady=(5, 10))
        
        # Registered faces
        faces_frame = tk.Frame(info_frame, bg='#34495e')
        faces_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            faces_frame,
            text="Registered Faces:",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(anchor=tk.W)
        
        self.registered_faces_label = tk.Label(
            faces_frame,
            text="0 faces registered",
            font=('Arial', 10),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.registered_faces_label.pack(anchor=tk.W, pady=(5, 10))
        
        # System info
        system_frame = tk.Frame(info_frame, bg='#34495e')
        system_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            system_frame,
            text="System Performance:",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(anchor=tk.W)
        
        self.fps_label = tk.Label(
            system_frame,
            text="FPS: 0.0",
            font=('Arial', 10),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.fps_label.pack(anchor=tk.W, pady=(5, 2))
        
        self.cpu_label = tk.Label(
            system_frame,
            text="CPU: 0.0%",
            font=('Arial', 10),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.cpu_label.pack(anchor=tk.W, pady=2)
        
        self.memory_label = tk.Label(
            system_frame,
            text="Memory: 0.0%",
            font=('Arial', 10),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.memory_label.pack(anchor=tk.W, pady=(2, 10))
        
        # Instructions
        instructions_frame = tk.Frame(info_frame, bg='#34495e')
        instructions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            instructions_frame,
            text="Instructions:",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(anchor=tk.W)
        
        instructions_text = """
• Click 'Login' to begin
• Camera window will open separately
• Press 'R' key to register a detected face
• Enter a name when prompted
• System will automatically recognize registered faces
• Hold face for 3 seconds to complete login
• When login is successful, camera continues running without face recognition
• Click 'Stop' to end recognition and close camera
        """
        
        instructions_label = tk.Label(
            instructions_frame,
            text=instructions_text,
            font=('Arial', 9),
            bg='#34495e',
            fg='#bdc3c7',
            justify=tk.LEFT
        )
        instructions_label.pack(anchor=tk.W, pady=(5, 10))
        
        # Bind keyboard events
        self.root.bind('<Key>', self.handle_keypress)
        
    def toggle_recognition(self):
        """Toggle face recognition on/off."""
        if not self.is_running:
            self.start_recognition()
        else:
            self.stop_recognition()
    
    def start_recognition(self):
        """Start the face recognition system."""
        try:
            # Check if camera is already in use
            if self.inference:
                self.inference.cleanup()
                self.inference = None
                time.sleep(1)  # Wait a bit for cleanup
            
            self.inference = FaceRecognitionInference()
            if not self.inference.initialize():
                messagebox.showerror("Error", "Failed to initialize face recognition system. Camera might be in use by another application.")
                return
            
            self.is_running = True
            self.recognition_completed = False
            self.recognition_start_time = None
            self.current_user = None
            self.start_button.config(text="Stop Login", bg='#e74c3c')
            self.register_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Face Recognition Running")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recognition: {str(e)}")
    
    def stop_recognition(self):
        """Stop the face recognition system."""
        self.is_running = False
        self.recognition_completed = False
        self.start_button.config(text="Login", bg='#27ae60')
        self.register_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        
        # Close camera window
        cv2.destroyAllWindows()
        
        if self.inference:
            self.inference.cleanup()
            self.inference = None
            time.sleep(0.5)  # Wait for cleanup to complete
    
    def camera_loop(self):
        """Main camera processing loop."""
        last_update = time.time()
        frame_count = 0
        
        while self.is_running:
            try:
                frame = self.inference.capture_frame()
                if frame is not None:
                    if not self.recognition_completed:
                        # Process frame for face recognition
                        face_result, recognition_result = self.inference.process_frame(frame)
                        
                        # Check if recognition is completed
                        if recognition_result:
                            best_match, best_similarity = recognition_result
                            if best_match and best_similarity > 0.6:
                                # Start timing if not already started
                                if self.recognition_start_time is None:
                                    self.recognition_start_time = time.time()
                                    self.current_user = best_match
                                
                                # Check if recognition has been maintained for required duration
                                elapsed_time = time.time() - self.recognition_start_time
                                if elapsed_time >= self.recognition_duration:
                                    self.recognition_completed = True
                                    self.on_recognition_success(best_match, best_similarity)
                            else:
                                # Reset timing if recognition is lost
                                self.recognition_start_time = None
                                self.current_user = None
                    else:
                        # After login success, just show camera feed without face recognition
                        face_result = None
                        recognition_result = None
                    
                    # Update camera display
                    self.update_camera_display(frame, face_result, recognition_result)
                    
                    # Update system info every second
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_update >= 1.0:
                        self.update_system_info()
                        last_update = current_time
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                time.sleep(0.1)
    
    def update_camera_display(self, frame, face_result, recognition_result):
        """Update the camera display with current frame."""
        try:
            # Draw face detection and recognition results
            if face_result:
                x, y, w, h, confidence = face_result
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if recognition_result:
                    best_match, best_similarity = recognition_result
                    if best_match and best_similarity > 0.6:
                        cv2.putText(frame, f"{best_match}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Sim: {best_similarity:.2f}", (x, y+h+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add system info to frame
            height, width = frame.shape[:2]
            if self.recognition_completed:
                cv2.putText(frame, "Login Successful - Camera Active", (10, height-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'Q' to quit", (10, height-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Press 'R' to register face", (10, height-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'Q' to quit", (10, height-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame in separate window
            cv2.imshow('Raspberry Pi Face Recognition', frame)
            
            # Handle OpenCV window events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self.stop_recognition()
            elif (key == ord('r') or key == ord('R')) and not self.recognition_completed:
                self.register_face()
            
            # Update recognition label
            if self.recognition_completed:
                self.recognition_label.config(
                    text="Login Successful - Camera Active",
                    fg='#27ae60'
                )
            elif recognition_result:
                best_match, best_similarity = recognition_result
                if best_match and best_similarity > 0.6:
                    if self.recognition_start_time is not None:
                        elapsed_time = time.time() - self.recognition_start_time
                        remaining_time = max(0, self.recognition_duration - elapsed_time)
                        self.recognition_label.config(
                            text=f"{best_match} (Similarity: {best_similarity:.2f}) - Hold for {remaining_time:.1f}s",
                            fg='#27ae60'
                        )
                    else:
                        self.recognition_label.config(
                            text=f"{best_match} (Similarity: {best_similarity:.2f})",
                            fg='#27ae60'
                        )
                else:
                    self.recognition_label.config(
                        text="Unknown face detected",
                        fg='#e74c3c'
                    )
            else:
                self.recognition_label.config(
                    text="No face detected",
                    fg='#ecf0f1'
                )
                
        except Exception as e:
            print(f"Display update error: {e}")
    
    def update_system_info(self):
        """Update system performance information."""
        try:
            if self.inference:
                fps = self.inference.get_fps()
                cpu_percent = self.inference.get_cpu_percent()
                memory_percent = self.inference.get_memory_percent()
                
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.cpu_label.config(text=f"CPU: {cpu_percent:.1f}%")
                self.memory_label.config(text=f"Memory: {memory_percent:.1f}%")
                
                face_count = self.inference.get_face_count()
                self.registered_faces_label.config(text=f"{face_count} faces registered")
                
        except Exception as e:
            print(f"System info update error: {e}")
    
    def register_face(self):
        """Register a new face."""
        if not self.is_running or not self.inference:
            messagebox.showwarning("Warning", "Please start face recognition first")
            return
        
        # Get the current face result
        current_face_result = self.inference.get_current_face_result()
        if not current_face_result:
            messagebox.showinfo("Info", "No face detected. Please position your face in front of the camera.")
            return
        
        # Prompt for name
        name = simpledialog.askstring("Register Face", "Enter name for the detected face:")
        if name and name.strip():
            name = name.strip()
            
            # Check if name already exists
            if self.inference.is_face_registered(name):
                result = messagebox.askyesno(
                    "Name Exists", 
                    f"Face '{name}' is already registered. Do you want to update it?"
                )
                if not result:
                    return
            
            # Register the face
            success = self.inference.register_face(name, current_face_result)
            if success:
                messagebox.showinfo("Success", f"Face '{name}' registered successfully!")
                self.update_system_info()  # Update face count
            else:
                messagebox.showerror("Error", "Failed to register face. Please try again.")
    
    def on_recognition_success(self, name, similarity):
        """Handle successful face recognition."""
        try:
            # Stop face recognition but keep camera running
            self.recognition_completed = True
            self.current_user = name
            
            # Show success message
            messagebox.showinfo("Login Successful", f"Welcome, {name}! (Similarity: {similarity:.2f})")
            
            # Transform current window to logged-in state
            self.transform_to_logged_in_screen(name)
            
        except Exception as e:
            print(f"Recognition success error: {e}")
    
    def transform_to_logged_in_screen(self, user_name):
        """Transform the current window to logged-in state."""
        try:
            # Clear the current window
            for widget in self.root.winfo_children():
                widget.destroy()
            
            # Update window title
            self.root.title(f"Welcome - {user_name}")
            self.root.geometry("800x600")
            
            # Main frame
            main_frame = tk.Frame(self.root, bg='#2c3e50')
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Top bar with logout button
            top_bar = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
            top_bar.pack(fill=tk.X, pady=(0, 20))
            
            # Logout button
            logout_button = tk.Button(
                top_bar,
                text="Logout",
                command=self.logout,
                font=('Arial', 12, 'bold'),
                bg='#e74c3c',
                fg='white',
                relief=tk.RAISED,
                bd=3,
                padx=20,
                pady=5
            )
            logout_button.pack(side=tk.RIGHT, padx=10, pady=10)
            
            # User info
            user_label = tk.Label(
                top_bar,
                text=f"Logged in as: {user_name}",
                font=('Arial', 12),
                bg='#34495e',
                fg='white'
            )
            user_label.pack(side=tk.LEFT, padx=10, pady=10)
            
            # Welcome message
            welcome_label = tk.Label(
                main_frame,
                text=f"Welcome, {user_name}!",
                font=('Arial', 24, 'bold'),
                bg='#2c3e50',
                fg='#27ae60'
            )
            welcome_label.pack(pady=(50, 30))
            
            # Success message
            success_label = tk.Label(
                main_frame,
                text="Login Successful",
                font=('Arial', 16),
                bg='#2c3e50',
                fg='white'
            )
            success_label.pack(pady=(0, 50))
            
            # System options frame
            options_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
            options_frame.pack(fill=tk.BOTH, expand=True, pady=20)
            
            options_title = tk.Label(
                options_frame,
                text="System Options",
                font=('Arial', 16, 'bold'),
                bg='#34495e',
                fg='white'
            )
            options_title.pack(pady=20)
            
            # Option buttons
            tk.Button(
                options_frame,
                text="Continue to System",
                command=self.continue_to_system,
                font=('Arial', 14, 'bold'),
                bg='#27ae60',
                fg='white',
                relief=tk.RAISED,
                bd=3,
                padx=30,
                pady=15
            ).pack(pady=20)
            
            # Don't close camera window - keep it running
            
        except Exception as e:
            print(f"Transform to logged-in screen error: {e}")
    
    def continue_to_system(self):
        """Continue to the main system."""
        messagebox.showinfo("System Access", "You now have access to the system!")
        # Add your system logic here
    
    def logout(self):
        """Logout and return to face recognition."""
        try:
            # Reset recognition state
            self.recognition_completed = False
            self.recognition_start_time = None
            self.current_user = None
            
            # Stop camera and close window
            self.is_running = False
            cv2.destroyAllWindows()
            
            # Recreate the original GUI
            self.setup_gui()
            
            # Update window title and size
            self.root.title("Face Recognition System")
            self.root.geometry("600x500")
            
            # Center the window
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f"{width}x{height}+{x}+{y}")
            
            messagebox.showinfo("Logout", "Logged out successfully. You can start face recognition again.")
            
        except Exception as e:
            print(f"Logout error: {e}")
    
    def handle_keypress(self, event):
        """Handle keyboard events."""
        if event.char.lower() == 'r' and self.is_running:
            self.register_face()
    
    def on_closing(self):
        """Handle application closing."""
        if self.is_running:
            self.stop_recognition()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    
    # Set closing protocol
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main() 
