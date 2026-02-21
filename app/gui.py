import tkinter as tk
import customtkinter as ctk
import subprocess
import sys
import os
import threading
from app.config import ACTIONS, ACTIONS_FILE

# Set look and feel
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SIBIGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SIBI Platform - Unified Interface")
        self.geometry("1100x700")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SIBI PLATFORM", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_recognition = ctk.CTkButton(self.sidebar_frame, text="Real-time Recognition", command=lambda: self.select_tab("Recognition"))
        self.btn_recognition.grid(row=1, column=0, padx=20, pady=10)

        self.btn_dataset = ctk.CTkButton(self.sidebar_frame, text="Dataset Management", command=lambda: self.select_tab("Dataset"))
        self.btn_dataset.grid(row=2, column=0, padx=20, pady=10)

        self.btn_training = ctk.CTkButton(self.sidebar_frame, text="Model Training", command=lambda: self.select_tab("Training"))
        self.btn_training.grid(row=3, column=0, padx=20, pady=10)

        self.btn_settings = ctk.CTkButton(self.sidebar_frame, text="System Settings", command=lambda: self.select_tab("Settings"))
        self.btn_settings.grid(row=4, column=0, padx=20, pady=10)

        # Create main content area
        self.content_frame = ctk.CTkFrame(self, corner_radius=10)
        self.content_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)

        # Console/Log area
        self.console_frame = ctk.CTkFrame(self, height=200)
        self.console_frame.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="nsew")
        self.console_frame.grid_columnconfigure(0, weight=1)
        self.console_frame.grid_rowconfigure(0, weight=1)

        self.console_text = ctk.CTkTextbox(self.console_frame, wrap="word")
        self.console_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.console_text.configure(state="disabled")

        self.current_tab = None
        self.select_tab("Recognition")

    def log(self, text):
        self.console_text.configure(state="normal")
        self.console_text.insert("end", text + "\n")
        self.console_text.see("end")
        self.console_text.configure(state="disabled")

    def run_command(self, cmd_args):
        """Run a command in a separate thread and capture output."""
        def target():
            self.log(f"> Executing: {' '.join(cmd_args)}")
            process = subprocess.Popen(
                [sys.executable, "-m"] + cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in process.stdout:
                self.log(line.strip())
            process.wait()
            self.log(f"--- Command finished with exit code {process.returncode} ---")

        threading.Thread(target=target, daemon=True).start()

    def select_tab(self, tab_name):
        if self.current_tab == tab_name:
            return
        
        # Clear current content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        self.current_tab = tab_name
        
        if tab_name == "Recognition":
            self.show_recognition_tab()
        elif tab_name == "Dataset":
            self.show_dataset_tab()
        elif tab_name == "Training":
            self.show_training_tab()
        elif tab_name == "Settings":
            self.show_settings_tab()

    def get_available_models(self):
        """Scan the models directory for available .keras and .tflite files."""
        from app.config import MODEL_PATH
        models = []
        if os.path.exists(MODEL_PATH):
            for f in os.listdir(MODEL_PATH):
                if f.endswith(".keras"):
                    models.append(f)
        return models or ["No models found"]

    def show_recognition_tab(self):
        title = ctk.CTkLabel(self.content_frame, text="Real-time Recognition", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=10)

        # Modality selection
        modality_frame = ctk.CTkFrame(self.content_frame)
        modality_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(modality_frame, text="Modality:").pack(side="left", padx=10)
        self.rec_modality = ctk.CTkSegmentedButton(modality_frame, values=["camera", "glove"])
        self.rec_modality.set("camera")
        self.rec_modality.pack(side="left", padx=10)

        # Model Selection
        model_frame = ctk.CTkFrame(self.content_frame)
        model_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(model_frame, text="Select Model:").pack(side="left", padx=10)
        available_models = self.get_available_models()
        self.model_dropdown = ctk.CTkComboBox(model_frame, values=available_models, width=300)
        self.model_dropdown.set(available_models[0])
        self.model_dropdown.pack(side="left", padx=10)

        btn_refresh = ctk.CTkButton(model_frame, text="↻", width=30, command=self.refresh_model_list)
        btn_refresh.pack(side="left", padx=5)

        btn_start = ctk.CTkButton(self.content_frame, text="START INFERENCE", fg_color="#2ecc71", hover_color="#27ae60", 
                                  font=ctk.CTkFont(size=14, weight="bold"),
                                  command=lambda: self.run_command(["app.main", "--mode", self.rec_modality.get(), "--model", self.model_dropdown.get()]))
        btn_start.grid(row=3, column=0, padx=20, pady=30, ipadx=20, ipady=10)

    def refresh_model_list(self):
        new_models = self.get_available_models()
        self.model_dropdown.configure(values=new_models)
        if new_models:
            self.model_dropdown.set(new_models[0])

    def show_dataset_tab(self):
        title = ctk.CTkLabel(self.content_frame, text="Dataset Management", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=10)

        # Action List
        info_frame = ctk.CTkFrame(self.content_frame)
        info_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        actions_list = ", ".join(ACTIONS)
        lbl_actions = ctk.CTkLabel(info_frame, text=f"Actions: {actions_list}", wraplength=700, justify="left")
        lbl_actions.pack(padx=20, pady=10)

        controls_frame = ctk.CTkFrame(self.content_frame)
        controls_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.new_action_entry = ctk.CTkEntry(controls_frame, placeholder_text="Action name (e.g. 'halo')")
        self.new_action_entry.pack(side="left", padx=10, expand=True, fill="x")

        btn_add = ctk.CTkButton(controls_frame, text="Add Action", command=self.on_add_action)
        btn_add.pack(side="left", padx=10)

        # Collection Section
        collect_frame = ctk.CTkFrame(self.content_frame)
        collect_frame.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        ctk.CTkLabel(collect_frame, text="Data Acquisition", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=3, pady=10)
        
        ctk.CTkLabel(collect_frame, text="Action:").grid(row=1, column=0, padx=10, pady=5)
        self.collect_action = ctk.CTkComboBox(collect_frame, values=list(ACTIONS))
        self.collect_action.grid(row=1, column=1, padx=10, pady=5)

        ctk.CTkLabel(collect_frame, text="Samples:").grid(row=2, column=0, padx=10, pady=5)
        self.collect_samples = ctk.CTkEntry(collect_frame, width=100)
        self.collect_samples.insert(0, "10")
        self.collect_samples.grid(row=2, column=1, padx=10, pady=5)

        btn_collect = ctk.CTkButton(collect_frame, text="Record Samples", fg_color="#3498db",
                                   command=lambda: self.run_command(["app.training.collect_data", "--action", self.collect_action.get(), "--samples", self.collect_samples.get()]))
        btn_collect.grid(row=1, column=2, rowspan=2, padx=20, pady=10, sticky="nsew")

    def on_add_action(self):
        action = self.new_action_entry.get()
        if action:
            self.run_command(["app.training.manage_dataset", "--add", action])
            # Note: The global ACTIONS won't update until restart unless we reload it
            self.new_action_entry.delete(0, 'end')

    def show_training_tab(self):
        title = ctk.CTkLabel(self.content_frame, text="Model Training", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=10)

        train_frame = ctk.CTkFrame(self.content_frame)
        train_frame.grid(row=1, column=0, padx=20, pady=20)

        ctk.CTkLabel(train_frame, text="Select Target Modality:").pack(pady=10)
        self.train_modality = ctk.CTkSegmentedButton(train_frame, values=["camera", "glove"])
        self.train_modality.set("camera")
        self.train_modality.pack(padx=20, pady=10)

        btn_train = ctk.CTkButton(self.content_frame, text="START TRAINING SESSION", fg_color="#e67e22", hover_color="#d35400",
                                 font=ctk.CTkFont(weight="bold"),
                                 command=lambda: self.run_command(["app.training.train_" + self.train_modality.get()]))
        btn_train.grid(row=2, column=0, padx=20, pady=20, ipadx=20, ipady=10)

    def show_settings_tab(self):
        title = ctk.CTkLabel(self.content_frame, text="System Settings", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=20)
        
        info_frame = ctk.CTkFrame(self.content_frame)
        info_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        lbl_info = ctk.CTkLabel(info_frame, text="Project Environment Information", font=ctk.CTkFont(weight="bold"))
        lbl_info.pack(padx=20, pady=10)

        ctk.CTkLabel(info_frame, text=f"Python: {sys.version.split()[0]}").pack(anchor="w", padx=20)
        ctk.CTkLabel(info_frame, text=f"Root: {os.getcwd()}").pack(anchor="w", padx=20)

        btn_gpu = ctk.CTkButton(self.content_frame, text="Run GPU Diagnostics", 
                                command=lambda: self.run_command(["app.scripts.util.test_gpu"]))
        btn_gpu.grid(row=2, column=0, padx=20, pady=20)


if __name__ == "__main__":
    app = SIBIGUI()
    app.mainloop()
