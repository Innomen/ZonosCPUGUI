#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import datetime

# ----------------------------
# Virtual Environment Handling
# ----------------------------
def in_virtualenv():
    # Check if we are running in a virtual environment.
    return (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

def ensure_virtualenv():
    if not in_virtualenv():
        print("Not running in a virtual environment. Creating one...")
        venv_dir = Path("venv")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        # Re-run the script using the venv python.
        if os.name == "nt":
            venv_python = str(venv_dir / "Scripts" / "python.exe")
        else:
            venv_python = str(venv_dir / "bin" / "python")
        print(f"Restarting with virtual environment interpreter: {venv_python}")
        os.execv(venv_python, [venv_python] + sys.argv)

# Ensure we are running in a virtual environment.
ensure_virtualenv()

# ----------------------------
# UV-Based Environment Setup
# ----------------------------
def is_uv_installed():
    try:
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def install_uv():
    print("Installing uv...")
    subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)

def run_uv_sync():
    """
    Run the uv commands as per the Zonos README:
      uv sync
      uv sync --extra compile
      uv pip install -e .[compile]
    These commands are run only once; we mark completion with an environment variable.
    """
    print("Running uv sync routines...")
    subprocess.run(["uv", "sync"], check=True)
    subprocess.run(["uv", "sync", "--extra", "compile"], check=True)
    subprocess.run(["uv", "pip", "install", "-e", ".[compile]"], check=True)
    os.environ["UV_SYNC_DONE"] = "1"
    print("uv sync complete; restarting script.")
    print(f"Current interpreter: {sys.executable}")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Run the uv sync routines if not already done.
if not os.environ.get("UV_SYNC_DONE"):
    if not is_uv_installed():
        install_uv()
    try:
        run_uv_sync()
    except Exception as e:
        print(f"uv sync failed: {e}")
        sys.exit(1)

# ----------------------------
# Begin GUI Front End Code
# ----------------------------
# At this point, we assume the virtual environment is set up correctly.
try:
    import torch
    import torchaudio
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print("It appears that some dependencies are missing. Please ensure that uv has installed torch correctly.")
    sys.exit(1)

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl, QSettings
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QTextEdit, QPushButton, QFileDialog,
    QMessageBox, QHBoxLayout, QComboBox, QLineEdit
)
from PyQt5.QtGui import QDesktopServices, QFont

# Default configuration – user settings are preserved using QSettings.
DEFAULT_CONFIG = {
    "output_dir": str(Path.home() / ".zonos_output"),
    "presets_dir": str(Path.home() / ".zonos_presets"),
    "ffmpeg_path": "ffmpeg",
    "timeout_seconds": 30
}

def force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("DRI_PRIME", None)
    torch.set_num_threads(1)

class MediaHandler:
    @staticmethod
    def load_audio(file_path, sample_rate=24000, timeout=30):
        try:
            wav, sr = torchaudio.load(file_path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            return wav
        except Exception as e:
            print(f"Torchaudio failed, using FFmpeg fallback: {str(e)}")
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(suffix=".wav") as tmpfile:
                cmd = [
                    "ffmpeg",
                    "-y", "-i", str(file_path),
                    "-ac", "1", "-ar", str(sample_rate),
                    "-vn", "-f", "wav", tmpfile.name
                ]
                subprocess.run(cmd, timeout=timeout, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return torchaudio.load(tmpfile.name)[0]

def scan_preset_voices(presets_dir):
    voices = {"Custom Voice": None}
    presets_path = Path(presets_dir)
    presets_path.mkdir(parents=True, exist_ok=True)
    for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac", "*.mp4", "*.mov", "*.mkv", "*.webm"]:
        for file in presets_path.glob(ext):
            voices[file.stem] = str(file.absolute())
    return voices

class GenerationThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    def __init__(self, model, audio_path, text, output_dir, config):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
        self.text = text
        self.output_dir = output_dir
        self.config = config
    def run(self):
        try:
            from zonos.conditioning import make_cond_dict
            wav = MediaHandler.load_audio(
                self.audio_path,
                sample_rate=24000,
                timeout=self.config["timeout_seconds"]
            )
            speaker = self.model.make_speaker_embedding(wav, 24000)
            cond_dict = make_cond_dict(text=self.text, speaker=speaker, language="en-us")
            codes = self.model.generate(self.model.prepare_conditioning(cond_dict))
            wavs = self.model.autoencoder.decode(codes).cpu()
            output_dir = Path(self.config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"output_{timestamp}.wav"
            torchaudio.save(output_file, wavs[0], self.model.autoencoder.sampling_rate)
            self.finished.emit(str(output_file.absolute()))
        except Exception as e:
            self.error.emit(str(e))

class VoiceCloneGUI(QMainWindow):
    def __init__(self, config, save_config_callback):
        super().__init__()
        self.config = config
        self.save_config = save_config_callback
        self.setWindowTitle("Zonos Voice Cloner")
        self.setGeometry(100, 100, 700, 600)
        font = QFont()
        font.setStyleHint(QFont.SansSerif)
        font.setPointSize(10)
        self.setFont(font)
        self.model = None
        self.current_voice_file = None
        self.last_output = None
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        self.setup_directory_ui(layout)
        self.setup_voice_ui(layout)
        layout.addWidget(QLabel("Text to Clone:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to convert to speech...")
        layout.addWidget(self.text_input)
        self.generate_btn = QPushButton("Generate Cloned Voice")
        self.generate_btn.clicked.connect(self.generate_speech)
        self.generate_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(self.generate_btn)
        layout.addWidget(QLabel("Latest Output:"))
        self.output_link = QLabel("No output generated yet")
        self.output_link.setStyleSheet("color: #0066cc; text-decoration: underline;")
        self.output_link.setCursor(Qt.PointingHandCursor)
        self.output_link.mousePressEvent = self.open_output_file
        layout.addWidget(self.output_link)
        self.status_bar = QLabel("Ready")
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_bar)
        self.load_model()
    def setup_directory_ui(self, layout):
        layout.addWidget(QLabel("Folder Settings:"))
        presets_layout = QHBoxLayout()
        presets_layout.addWidget(QLabel("Presets Directory:"))
        self.presets_dir_input = QLineEdit(self.config["presets_dir"])
        presets_layout.addWidget(self.presets_dir_input)
        self.presets_browse_btn = QPushButton("Browse...")
        self.presets_browse_btn.clicked.connect(lambda: self.browse_directory("presets_dir"))
        presets_layout.addWidget(self.presets_browse_btn)
        self.presets_open_btn = QPushButton("Open")
        self.presets_open_btn.clicked.connect(lambda: self.open_directory(self.config["presets_dir"]))
        presets_layout.addWidget(self.presets_open_btn)
        layout.addLayout(presets_layout)
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_input = QLineEdit(self.config["output_dir"])
        output_layout.addWidget(self.output_dir_input)
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(lambda: self.browse_directory("output_dir"))
        output_layout.addWidget(self.output_browse_btn)
        self.output_open_btn = QPushButton("Open")
        self.output_open_btn.clicked.connect(lambda: self.open_directory(self.config["output_dir"]))
        output_layout.addWidget(self.output_open_btn)
        layout.addLayout(output_layout)
        self.save_btn = QPushButton("Save Folder Settings")
        self.save_btn.clicked.connect(self.save_folder_settings)
        layout.addWidget(self.save_btn)
    def setup_voice_ui(self, layout):
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Select Voice:"))
        self.voice_combo = QComboBox()
        self.refresh_voice_list()
        self.voice_combo.currentTextChanged.connect(self.handle_voice_change)
        voice_layout.addWidget(self.voice_combo)
        self.play_btn = QPushButton("Play Original")
        self.play_btn.clicked.connect(self.play_voice_preview)
        self.play_btn.setToolTip("Play the original voice file")
        voice_layout.addWidget(self.play_btn)
        layout.addLayout(voice_layout)
        self.custom_voice_layout = QHBoxLayout()
        self.custom_voice_layout.addWidget(QLabel("Custom Voice File:"))
        self.voice_path_label = QLabel("No file selected")
        self.voice_path_label.setStyleSheet("color: #666; font-style: italic;")
        self.custom_voice_layout.addWidget(self.voice_path_label)
        self.custom_voice_layout.addStretch()
        self.browse_voice_btn = QPushButton("Browse...")
        self.browse_voice_btn.clicked.connect(self.select_voice_file)
        self.browse_voice_btn.setToolTip("Supports: Audio (MP3/WAV/OGG/FLAC) or Video (MP4/MOV/MKV) files")
        self.custom_voice_layout.addWidget(self.browse_voice_btn)
        self.custom_voice_layout.setEnabled(False)
        layout.addLayout(self.custom_voice_layout)
    def refresh_voice_list(self):
        self.voice_combo.clear()
        def scan_preset_voices(presets_dir):
            voices = {"Custom Voice": None}
            presets_path = Path(presets_dir)
            presets_path.mkdir(parents=True, exist_ok=True)
            for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac", "*.mp4", "*.mov", "*.mkv", "*.webm"]:
                for file in presets_path.glob(ext):
                    voices[file.stem] = str(file.absolute())
            return voices
        self.available_voices = scan_preset_voices(self.config["presets_dir"])
        self.voice_combo.addItems(self.available_voices.keys())
        if "Custom Voice" not in self.available_voices:
            self.voice_combo.addItem("Custom Voice")
    def browse_directory(self, dir_type):
        from PyQt5.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, f"Select {dir_type.replace('_', ' ').title()}")
        if dir_path:
            if dir_type == "presets_dir":
                self.presets_dir_input.setText(dir_path)
            else:
                self.output_dir_input.setText(dir_path)
    def open_directory(self, path):
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        if Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.warning(self, "Error", f"Directory not found:\n{path}")
    def save_folder_settings(self):
        from PyQt5.QtCore import QSettings
        settings = QSettings("Zonos", "VoiceCloner")
        self.config["presets_dir"] = self.presets_dir_input.text()
        self.config["output_dir"] = self.output_dir_input.text()
        for key, value in self.config.items():
            settings.setValue(key, value)
        self.refresh_voice_list()
        QMessageBox.information(self, "Success", "Folder settings saved and updated")
    def handle_voice_change(self, voice_name):
        if voice_name == "Custom Voice":
            self.custom_voice_layout.setEnabled(True)
            self.current_voice_file = None
            self.voice_path_label.setText("No file selected")
            self.play_btn.setEnabled(False)
        else:
            self.custom_voice_layout.setEnabled(False)
            voice_file = self.available_voices.get(voice_name)
            if voice_file and Path(voice_file).exists():
                self.current_voice_file = Path(voice_file)
                self.voice_path_label.setText(str(self.current_voice_file))
                self.play_btn.setEnabled(True)
            else:
                self.current_voice_file = None
                self.voice_path_label.setText("Preset file not found")
                self.play_btn.setEnabled(False)
    def select_voice_file(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Voice File",
            "",
            "Media Files (*.wav *.mp3 *.ogg *.flac *.mp4 *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.current_voice_file = Path(file_path)
            self.voice_path_label.setText(file_path)
            if file_path not in self.available_voices.values():
                self.refresh_voice_list()
            self.play_btn.setEnabled(True)
    def play_voice_preview(self):
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        voice_name = self.voice_combo.currentText()
        if voice_name == "Custom Voice":
            if self.current_voice_file:
                self.play_audio_file(self.current_voice_file)
            else:
                QMessageBox.warning(self, "Error", "No custom voice file selected")
            return
        voice_file = self.available_voices.get(voice_name)
        if not voice_file:
            QMessageBox.warning(self, "Error", "No voice file found for selected preset")
            return
        if not Path(voice_file).exists():
            QMessageBox.warning(self, "Error", f"Voice file not found:\n{voice_file}")
            return
        self.play_audio_file(voice_file)
    def play_audio_file(self, file_path):
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))
            self.status_bar.setText(f"Playing: {Path(file_path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Couldn't play file:\n{str(e)}")
    def open_output_file(self, event):
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        if self.last_output and Path(self.last_output).exists():
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Couldn't open file:\n{str(e)}")
        else:
            QMessageBox.warning(self, "Error", f"Output file not found:\n{self.last_output}")
    def load_model(self):
        try:
            force_cpu()
            from zonos.model import Zonos
            self.status_bar.setText("Loading model...")
            QApplication.processEvents()
            self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu")
            self.status_bar.setText("Model loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.status_bar.setText("Model loading failed")
    def generate_speech(self):
        if not self.model:
            QMessageBox.warning(self, "Error", "Model not loaded")
            return
        if not self.current_voice_file:
            QMessageBox.warning(self, "Error", "Please select a voice file")
            return
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Please enter some text")
            return
        self.set_ui_enabled(False)
        self.status_bar.setText("Generating speech...")
        from pathlib import Path
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.thread = GenerationThread(
            model=self.model,
            audio_path=self.current_voice_file,
            text=text,
            output_dir=output_dir,
            config=self.config
        )
        self.thread.finished.connect(self.generation_complete)
        self.thread.error.connect(self.generation_error)
        self.thread.start()
    def set_ui_enabled(self, enabled):
        self.voice_combo.setEnabled(enabled)
        self.browse_voice_btn.setEnabled(enabled)
        self.text_input.setEnabled(enabled)
        self.generate_btn.setEnabled(enabled)
        self.play_btn.setEnabled(enabled)
    def generation_complete(self, output_file):
        self.set_ui_enabled(True)
        self.last_output = output_file
        from pathlib import Path
        short_path = str(Path(output_file).relative_to(Path.home()))
        self.output_link.setText(f"Click to play: ~/{short_path}")
        self.status_bar.setText(f"Output saved to: {output_file}")
        QMessageBox.information(self, "Success", f"Output saved to:\n{output_file}")
    def generation_error(self, error_msg):
        self.set_ui_enabled(True)
        self.status_bar.setText("Generation failed")
        QMessageBox.critical(self, "Error", f"Generation failed:\n{error_msg}")

def main():
    from PyQt5.QtCore import QSettings
    app = QApplication(sys.argv)
    font = QFont()
    font.setStyleHint(QFont.SansSerif)
    font.setPointSize(10)
    app.setFont(font)
    settings = QSettings("Zonos", "VoiceCloner")
    config = DEFAULT_CONFIG.copy()
    for key in config:
        value = settings.value(key)
        if value:
            config[key] = value
    def save_config(config):
        for key, value in config.items():
            settings.setValue(key, value)
    window = VoiceCloneGUI(config, save_config)
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
