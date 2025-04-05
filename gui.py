#!/usr/bin/env python3
import os, sys, traceback
from pathlib import Path
import datetime

EXPECTED_VENV_DIR = Path.cwd() / ".venv"

def check_environment():
    # Check environment activated
    venv_path = os.getenv("VIRTUAL_ENV")
    if not venv_path or not venv_path.startswith(str(EXPECTED_VENV_DIR)):
        print("\n[Zonos Launcher] Error: Not in the dedicated virtual environment!")
        print(f"Expected VENV activated in: {EXPECTED_VENV_DIR}")
        print("Please run:\n")
        print("  source .venv/bin/activate")
        print("  python gui.py\n")
        sys.exit(1)

check_environment()

REQUIRED_PKGS = [
    "torch", "torchaudio", "numpy", "PyQt5", "transformers",
    "huggingface_hub", "sentencepiece", "safetensors", "gradio",
    "langcodes", "einops", "kanjize"
]

missing = []
for pkg in REQUIRED_PKGS:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print("\n==== Missing required packages ====\n")
    print(" ".join(missing))
    print("\nPlease run setup again:\n")
    print("  source .venv/bin/activate")
    print("  ./setup_zonos.sh\n")
    sys.exit(1)

# === Imports proceed safely ===

try:
    import torch, torchaudio
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QSettings
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import QFont, QDesktopServices
except Exception as e:
    print("\nUnexpected import error:", e)
    sys.exit(1)

# Optionally force CPU use only, safer on many systems
def force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("DRI_PRIME", None)
    torch.set_num_threads(1)

DEFAULT_CONFIG = {
    'output_dir': str(Path.home() / '.zonos_output'),
    'presets_dir': str(Path.home() / '.zonos_presets')
}

def scan_presets(pdir):
    presets = {'Custom Voice': None}
    Path(pdir).mkdir(parents=True, exist_ok=True)
    for ext in ['*.wav','*.mp3','*.ogg','*.flac','*.mp4','*.mov','*.mkv','*.webm']:
        for f in Path(pdir).glob(ext):
            presets[f.stem] = str(f.resolve())
    return presets

class Worker(QThread):
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, voice_path, text, outdir):
        super().__init__()
        self.model = model
        self.voice_path = voice_path
        self.text = text
        self.outdir = outdir

    def run(self):
        try:
            from zonos.conditioning import make_cond_dict
            wav, sr = torchaudio.load(self.voice_path)
            speaker = self.model.make_speaker_embedding(wav, sr)
            cond = make_cond_dict(text=self.text, speaker=speaker, language='en-us')
            prep = self.model.prepare_conditioning(cond)
            codes = self.model.generate(prep)
            wavs = self.model.autoencoder.decode(codes).cpu()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(self.outdir)/f"zonos_{ts}.wav"
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_path, wavs[0], self.model.autoencoder.sampling_rate)
            self.done.emit(str(out_path.resolve()))
        except Exception:
            self.error.emit(traceback.format_exc())

class VoiceCloner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zonos Voice Cloner")
        font = QFont(); font.setPointSize(9); self.setFont(font)

        self.settings = QSettings("Zonos", "VoiceCloner")
        self.config = dict(DEFAULT_CONFIG)
        for k in self.config:
            v = self.settings.value(k)
            if v: self.config[k] = v

        self.last_output = None

        cw = QWidget(); self.setCentralWidget(cw)
        lay = QVBoxLayout(cw)

        fr = QHBoxLayout()
        fr.addWidget(QLabel("Presets folder:"))
        self.presets_edit = QLineEdit(self.config['presets_dir']); fr.addWidget(self.presets_edit)
        pb = QPushButton("Browse"); pb.clicked.connect(self.set_presets_dir); fr.addWidget(pb)
        fo = QPushButton("Open"); fo.clicked.connect(lambda:self.open_dir(self.presets_edit.text())); fr.addWidget(fo)
        lay.addLayout(fr)

        ofr = QHBoxLayout()
        ofr.addWidget(QLabel("Output folder:"))
        self.out_edit = QLineEdit(self.config['output_dir']); ofr.addWidget(self.out_edit)
        pb2 = QPushButton("Browse"); pb2.clicked.connect(self.set_output_dir); ofr.addWidget(pb2)
        fo2 = QPushButton("Open"); fo2.clicked.connect(lambda:self.open_dir(self.out_edit.text())); ofr.addWidget(fo2)
        lay.addLayout(ofr)

        savebtn = QPushButton("Save folders"); savebtn.clicked.connect(self.save_paths)
        lay.addWidget(savebtn)

        hr = QHBoxLayout()
        hr.addWidget(QLabel("Voice Preset:"))
        self.combo = QComboBox(); hr.addWidget(self.combo)
        self.combo.currentTextChanged.connect(self.voice_change)
        self.playbtn = QPushButton("Play"); hr.addWidget(self.playbtn)
        self.playbtn.clicked.connect(self.play_voice)
        lay.addLayout(hr)

        self.custom_container = QWidget()
        ch = QHBoxLayout(self.custom_container)
        ch.addWidget(QLabel("Custom file:"))
        self.cust_label = QLabel("None"); ch.addWidget(self.cust_label)
        bc = QPushButton("Browse..."); ch.addWidget(bc)
        bc.clicked.connect(self.pick_custom)
        lay.addWidget(self.custom_container)
        self.custom_container.setVisible(False)

        lay.addWidget(QLabel("Text:"))
        self.textbox = QTextEdit(); lay.addWidget(self.textbox)

        self.genbtn = QPushButton("Generate voice"); lay.addWidget(self.genbtn)
        self.genbtn.clicked.connect(self.generate)

        self.status = QLabel("Ready"); lay.addWidget(self.status)
        self.outlabel = QLabel("No output yet."); lay.addWidget(self.outlabel)

        self.refresh_presets()
        self.model = None
        self.voice_path = None

        self.load_model()

    def refresh_presets(self):
        self.presets = scan_presets(self.presets_edit.text())
        self.combo.clear()
        self.combo.addItems(self.presets.keys())

    def save_paths(self):
        self.config['presets_dir'] = self.presets_edit.text()
        self.config['output_dir'] = self.out_edit.text()
        for k, v in self.config.items():
            self.settings.setValue(k, v)
        self.refresh_presets()
        QMessageBox.information(self, "Saved", "Folders saved.")

    def set_presets_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Presets Directory")
        if d:
            self.presets_edit.setText(d)
            self.refresh_presets()

    def set_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self.out_edit.setText(d)

    def open_dir(self, path):
        p = Path(path)
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    def voice_change(self, vname):
        if vname == "Custom Voice":
            self.custom_container.setVisible(True)
            self.voice_path = None
            self.cust_label.setText("None")
            self.playbtn.setEnabled(False)
            return
        else:
            self.custom_container.setVisible(False)
            f = self.presets.get(vname)
            if f and Path(f).exists():
                self.voice_path = Path(f)
                self.playbtn.setEnabled(True)
            else:
                self.voice_path = None
                self.playbtn.setEnabled(False)

    def pick_custom(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select audio")
        if f:
            self.voice_path = Path(f)
            self.cust_label.setText(f)
            self.playbtn.setEnabled(True)

    def play_voice(self):
        if self.voice_path and self.voice_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.voice_path.resolve())))

    def load_model(self):
        try:
            force_cpu()
            from zonos.model import Zonos
            self.status.setText("Loading model...")
            QApplication.processEvents()
            self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device='cpu')
            self.status.setText("Model loaded.")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            QMessageBox.critical(self, "Error loading model", tb)
            self.status.setText("Model load error")

    def generate(self):
        if not self.model:
            QMessageBox.warning(self, "No model", "Model not loaded yet")
            return
        txt = self.textbox.toPlainText().strip()
        if not txt:
            QMessageBox.warning(self, "No input", "Please enter some text")
            return
        if not self.voice_path or not self.voice_path.exists():
            QMessageBox.warning(self, "No voice", "Please select a reference voice")
            return

        self.status.setText("Generating...")
        QApplication.processEvents()
        wrk = Worker(self.model, self.voice_path, txt, self.out_edit.text())
        self.worker = wrk
        wrk.done.connect(self.gen_done)
        wrk.error.connect(self.gen_error)
        wrk.start()

    def gen_done(self, path):
        self.last_output = path
        self.status.setText("Done")
        self.outlabel.setText(f"Saved to:\n{path}")
        QMessageBox.information(self, "Done", f"Voice saved:\n{path}")

    def gen_error(self, error_text):
        self.status.setText("Error")
        print(error_text)
        QMessageBox.critical(self, "Generation failed", error_text)

def main():
    app = QApplication(sys.argv)
    font = QFont(); font.setPointSize(10); app.setFont(font)
    w = VoiceCloner(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
