#!/usr/bin/env python3
import os
import sys
import subprocess
import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch
import torchaudio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QTextEdit, QPushButton, QFileDialog,
    QMessageBox, QHBoxLayout, QComboBox, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QProcess, QSettings
from PyQt5.QtGui import QDesktopServices, QFont

# --- Configuration ---
DEFAULT_CONFIG = {
    "output_dir": os.path.expanduser("~/.zonos_output"),
    "presets_dir": os.path.expanduser("~/.zonos_presets"),
    "ffmpeg_path": "ffmpeg",
    "timeout_seconds": 30
}

class MediaHandler:
    @staticmethod
    def load_audio(file_path, sample_rate=24000, timeout=30):
        """Universal media loader with FFmpeg fallback"""
        try:
            wav, sr = torchaudio.load(file_path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            return wav
        except Exception as e:
            print(f"Torchaudio failed, using FFmpeg fallback: {str(e)}")
            with NamedTemporaryFile(suffix=".wav") as tmpfile:
                cmd = [
                    "ffmpeg",
                    "-y", "-i", str(file_path),
                    "-ac", "1", "-ar", str(sample_rate),
                    "-vn", "-f", "wav", tmpfile.name
                ]
                try:
                    subprocess.run(
                        cmd,
                        timeout=timeout,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    return torchaudio.load(tmpfile.name)[0]
                except subprocess.TimeoutExpired:
                    raise RuntimeError("FFmpeg timeout - file may be corrupt")
                except Exception as e:
                    raise RuntimeError(f"FFmpeg failed: {str(e)}")

def force_cpu():
    """Ensure CPU-only operation"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("DRI_PRIME", None)
    torch.set_num_threads(1)

def scan_preset_voices(presets_dir):
    """Scan presets directory for available voices"""
    voices = {"Custom Voice": None}
    presets_path = Path(presets_dir)
    presets_path.mkdir(parents=True, exist_ok=True)

    extensions = ["*.wav", "*.mp3", "*.ogg", "*.flac", "*.mp4", "*.mov", "*.mkv", "*.webm"]
    for ext in extensions:
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
        self.make_cond_dict = None

    def run(self):
        try:
            from zonos.conditioning import make_cond_dict
            self.make_cond_dict = make_cond_dict

            wav = MediaHandler.load_audio(
                self.audio_path,
                sample_rate=24000,
                timeout=self.config["timeout_seconds"]
            )

            speaker = self.model.make_speaker_embedding(wav, 24000)
            cond_dict = self.make_cond_dict(text=self.text, speaker=speaker, language="en-us")
            codes = self.model.generate(self.model.prepare_conditioning(cond_dict))
            wavs = self.model.autoencoder.decode(codes).cpu()

            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"output_{timestamp}.wav"
            torchaudio.save(output_file, wavs[0], self.model.autoencoder.sampling_rate)

            self.finished.emit(str(output_file.absolute()))
        except Exception as e:
            self.error.emit(str(e))

class VoiceCloneGUI(QMainWindow):
    def __init__(self, config, save_config_callback):
        super().__init__()
        self.config = config
        self.save_config = save_config_callback

        # Window setup
        self.setWindowTitle("Zonos Voice Cloner")
        self.setGeometry(100, 100, 700, 600)
        self.setFont(QFont("Sans Serif", 10))

        # Initialize
        self.model = None
        self.current_voice_file = None
        self.last_output = None

        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Directory settings
        self.setup_directory_ui(layout)

        # Voice selection
        self.setup_voice_ui(layout)

        # Text input
        layout.addWidget(QLabel("Text to Clone:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to convert to speech...")
        layout.addWidget(self.text_input)

        # Generate button
        self.generate_btn = QPushButton("Generate Cloned Voice")
        self.generate_btn.clicked.connect(self.generate_speech)
        self.generate_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(self.generate_btn)

        # Output section
        layout.addWidget(QLabel("Latest Output:"))
        self.output_link = QLabel("No output generated yet")
        self.output_link.setStyleSheet("color: #0066cc; text-decoration: underline;")
        self.output_link.setCursor(Qt.PointingHandCursor)
        self.output_link.mousePressEvent = self.open_output_file
        layout.addWidget(self.output_link)

        # Status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_bar)

        # Load model
        self.load_model()

    def setup_directory_ui(self, layout):
        """Setup directory configuration UI"""
        layout.addWidget(QLabel("Folder Settings:"))

        # Presets directory
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

        # Output directory
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

        # Save settings button
        self.save_btn = QPushButton("Save Folder Settings")
        self.save_btn.clicked.connect(self.save_folder_settings)
        layout.addWidget(self.save_btn)

    def setup_voice_ui(self, layout):
        """Setup voice selection UI"""
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Select Voice:"))

        self.voice_combo = QComboBox()
        self.refresh_voice_list()
        self.voice_combo.currentTextChanged.connect(self.handle_voice_change)
        voice_layout.addWidget(self.voice_combo)

        # Add play button for voice preview
        self.play_btn = QPushButton("Play Original")
        self.play_btn.clicked.connect(self.play_voice_preview)
        self.play_btn.setToolTip("Play the original voice file")
        voice_layout.addWidget(self.play_btn)

        layout.addLayout(voice_layout)

        # Custom voice file selection
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

    def play_voice_preview(self):
        """Play the original voice file for preview"""
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
        """Play audio file using default system player"""
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))
            self.status_bar.setText(f"Playing: {Path(file_path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Couldn't play file:\n{str(e)}")

    def browse_directory(self, dir_type):
        """Browse for a directory"""
        dir_path = QFileDialog.getExistingDirectory(self, f"Select {dir_type.replace('_', ' ').title()}")
        if dir_path:
            if dir_type == "presets_dir":
                self.presets_dir_input.setText(dir_path)
            else:
                self.output_dir_input.setText(dir_path)

    def open_directory(self, path):
        """Open directory in file manager"""
        if Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.warning(self, "Error", f"Directory not found:\n{path}")

    def save_folder_settings(self):
        """Save the folder settings"""
        self.config["presets_dir"] = self.presets_dir_input.text()
        self.config["output_dir"] = self.output_dir_input.text()
        self.save_config(self.config)
        self.refresh_voice_list()
        QMessageBox.information(self, "Success", "Folder settings saved and updated")

    def refresh_voice_list(self):
        """Refresh the list of available voices"""
        self.voice_combo.clear()
        self.available_voices = scan_preset_voices(self.config["presets_dir"])
        self.voice_combo.addItems(self.available_voices.keys())
        if "Custom Voice" not in self.available_voices:
            self.voice_combo.addItem("Custom Voice")

    def open_output_file(self, event):
        """Open the last output file"""
        if self.last_output and Path(self.last_output).exists():
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Couldn't open file:\n{str(e)}")
        else:
            QMessageBox.warning(self, "Error", f"Output file not found:\n{self.last_output}")

    def load_model(self):
        """Load the Zonos model"""
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

    def handle_voice_change(self, voice_name):
        """Handle voice selection changes"""
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
        """Select a custom voice file"""
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

    def generate_speech(self):
        """Generate cloned voice output"""
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
        """Toggle UI elements"""
        self.voice_combo.setEnabled(enabled)
        self.browse_voice_btn.setEnabled(enabled)
        self.text_input.setEnabled(enabled)
        self.generate_btn.setEnabled(enabled)
        self.play_btn.setEnabled(enabled)

    def generation_complete(self, output_file):
        """Handle successful generation"""
        self.set_ui_enabled(True)
        self.last_output = output_file
        short_path = str(Path(output_file).relative_to(Path.home()))
        self.output_link.setText(f"Click to play: ~/{short_path}")
        self.status_bar.setText(f"Output saved to: {output_file}")
        QMessageBox.information(self, "Success", f"Output saved to:\n{output_file}")

    def generation_error(self, error_msg):
        """Handle generation errors"""
        self.set_ui_enabled(True)
        self.status_bar.setText("Generation failed")
        QMessageBox.critical(self, "Error", f"Generation failed:\n{error_msg}")

def main():
    # Suppress warnings
    os.environ["QT_LOGGING_RULES"] = "*.warning=false"
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    app = QApplication(sys.argv)
    app.setFont(QFont("Sans Serif", 10))

    # Load settings
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
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
