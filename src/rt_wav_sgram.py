'''Realtime waveform and spectrogram

Features:
- Left sidebar with controllable parameters
- Real-time parameter updates (dB range, colormap, FPS)
- FPS tracker display

---
- 2026-02-13 jkang & claude created
'''
import sys
import time
import threading
import numpy as np
import sounddevice as sd

# Use scipy.fft if available (faster than numpy.fft)
try:
    from scipy import fft as scipy_fft
    USE_SCIPY_FFT = True
except ImportError:
    USE_SCIPY_FFT = False

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class RingBuffer:
    '''Thread-safe ring buffer for float32 mono audio.'''

    def __init__(self, size: int):
        self.size = int(size)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.write_idx = 0
        self.lock = threading.Lock()

    def write(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = x.size
        if n <= 0:
            return
        with self.lock:
            if n >= self.size:
                # keep only last 'size' samples
                self.buf[:] = x[-self.size:]
                self.write_idx = 0
                return

            end = self.write_idx + n
            if end < self.size:
                self.buf[self.write_idx:end] = x
            else:
                k = self.size - self.write_idx
                self.buf[self.write_idx:] = x[:k]
                self.buf[:end - self.size] = x[k:]
            self.write_idx = end % self.size

    def read_latest(self, n: int, out: np.ndarray = None) -> np.ndarray:
        '''
        Read the latest n samples in time order.

        Args:
            n: Number of samples to read
            out: Optional pre-allocated output array (float32, shape >= n)

        Returns:
            Array of n most recent samples in chronological order.
        '''
        n = int(n)
        n = min(n, self.size)

        if out is None:
            out = np.empty(n, dtype=np.float32)

        with self.lock:
            idx = self.write_idx
            start = (idx - n) % self.size

            if start < idx:
                out[:n] = self.buf[start:idx]
            else:
                k = self.size - start
                out[:k] = self.buf[start:]
                out[k:n] = self.buf[:idx]

        return out[:n]

    def resize(self, new_size: int):
        '''
        Resize the buffer while preserving existing data.

        If new_size > current size: old data is kept at the end (most recent)
        If new_size < current size: only the most recent samples are kept
        '''
        new_size = int(new_size)
        with self.lock:
            # Read all current data in order
            old_data = np.empty(self.size, dtype=np.float32)
            idx = self.write_idx
            if idx == 0:
                old_data[:] = self.buf
            else:
                old_data[:self.size - idx] = self.buf[idx:]
                old_data[self.size - idx:] = self.buf[:idx]

            # Create new buffer
            new_buf = np.zeros(new_size, dtype=np.float32)

            if new_size >= self.size:
                # Expanding: put old data at the end
                new_buf[new_size - self.size:] = old_data
            else:
                # Shrinking: keep only most recent samples
                new_buf[:] = old_data[self.size - new_size:]

            self.buf = new_buf
            self.size = new_size
            self.write_idx = 0


class ControlSidebar(QtWidgets.QWidget):
    '''Sidebar widget containing all parameter controls.'''

    # Signals for real-time parameters
    db_range_changed = QtCore.pyqtSignal(float, float)
    max_freq_changed = QtCore.pyqtSignal(int)
    colormap_changed = QtCore.pyqtSignal(str)
    fps_target_changed = QtCore.pyqtSignal(int)
    wave_gain_changed = QtCore.pyqtSignal(float)

    # Signals for stream parameters
    stream_settings_changed = QtCore.pyqtSignal(dict)
    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        '''Create all sidebar widgets.'''
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # Header
        header = QtWidgets.QLabel('Controls')
        header.setStyleSheet('font-size: 18px; font-weight: bold; color: #0066cc;')
        layout.addWidget(header)

        # Real-time parameters group
        layout.addWidget(self._create_realtime_group())

        # Stream parameters group
        layout.addWidget(self._create_stream_group())

        # Stream control group
        layout.addWidget(self._create_control_group())

        # Spacer
        layout.addStretch()

        # Note panel
        layout.addWidget(self._create_note_panel())

    def _create_realtime_group(self) -> QtWidgets.QGroupBox:
        '''Create real-time parameter controls.'''
        group = QtWidgets.QGroupBox('Display Settings')
        form = QtWidgets.QFormLayout(group)
        form.setSpacing(8)

        # dB Min (Praat-like: ~50 dB dynamic range)
        self.db_min_spin = QtWidgets.QDoubleSpinBox()
        self.db_min_spin.setRange(-120.0, -10.0)
        self.db_min_spin.setSingleStep(5.0)
        self.db_min_spin.setValue(-50.0)
        self.db_min_spin.setSuffix(' dB')
        form.addRow('dB Min:', self.db_min_spin)

        # dB Max
        self.db_max_spin = QtWidgets.QDoubleSpinBox()
        self.db_max_spin.setRange(-40.0, 20.0)
        self.db_max_spin.setSingleStep(5.0)
        self.db_max_spin.setValue(0.0)
        self.db_max_spin.setSuffix(' dB')
        form.addRow('dB Max:', self.db_max_spin)

        # Max Frequency
        self.max_freq_spin = QtWidgets.QSpinBox()
        self.max_freq_spin.setRange(1000, 24000)
        self.max_freq_spin.setSingleStep(500)
        self.max_freq_spin.setValue(6000)
        self.max_freq_spin.setSuffix(' Hz')
        form.addRow('Max Freq:', self.max_freq_spin)

        # Colormap (pyqtgraph built-in colormaps + custom grayscale)
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems([
            'grayscale', 'viridis', 'plasma', 'inferno', 'magma',
            'cividis', 'turbo', 'CET-L9'
        ])
        form.addRow('Colormap:', self.colormap_combo)

        # FPS Target
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(10, 120)
        self.fps_spin.setSingleStep(5)
        self.fps_spin.setValue(60)
        self.fps_spin.setSuffix(' FPS')
        form.addRow('Target FPS:', self.fps_spin)

        # Waveform Gain
        self.wave_gain_spin = QtWidgets.QDoubleSpinBox()
        self.wave_gain_spin.setRange(1.0, 100.0)
        self.wave_gain_spin.setSingleStep(1.0)
        self.wave_gain_spin.setValue(4.0)
        self.wave_gain_spin.setSuffix('x')
        form.addRow('Wave Gain:', self.wave_gain_spin)

        return group

    def _create_stream_group(self) -> QtWidgets.QGroupBox:
        '''Create stream parameter controls (restart required).'''
        group = QtWidgets.QGroupBox('Audio Stream')
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        # Note label
        note = QtWidgets.QLabel('Changes require Apply')
        note.setStyleSheet('color: #888; font-size: 11px; font-style: italic;')
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

        # Sample Rate
        self.sample_rate_combo = QtWidgets.QComboBox()
        for sr in [8000, 16000, 22050, 44100, 48000]:
            self.sample_rate_combo.addItem(f'{sr} Hz', sr)
        self.sample_rate_combo.setCurrentIndex(1)  # 16000
        form.addRow('Sample Rate:', self.sample_rate_combo)

        # Duration
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 10.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setValue(4.0)
        self.duration_spin.setSuffix(' s')
        form.addRow('Duration:', self.duration_spin)

        # Block Size
        self.blocksize_combo = QtWidgets.QComboBox()
        for bs in [128, 256, 512, 1024]:
            self.blocksize_combo.addItem(str(bs), bs)
        self.blocksize_combo.setCurrentIndex(1)  # 256
        form.addRow('Block Size:', self.blocksize_combo)

        # n_fft
        self.nfft_combo = QtWidgets.QComboBox()
        for nfft in [256, 512, 1024, 2048, 4096]:
            self.nfft_combo.addItem(str(nfft), nfft)
        self.nfft_combo.setCurrentIndex(2)  # 1024
        form.addRow('FFT Size:', self.nfft_combo)

        # Hop Size
        self.hop_combo = QtWidgets.QComboBox()
        for hop in [32, 64, 128, 256, 512]:
            self.hop_combo.addItem(str(hop), hop)
        self.hop_combo.setCurrentIndex(2)  # 128
        form.addRow('Hop Size:', self.hop_combo)

        layout.addLayout(form)

        # Apply button
        self.apply_btn = QtWidgets.QPushButton('Apply Changes')
        self.apply_btn.setStyleSheet('''
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
        ''')
        layout.addWidget(self.apply_btn)

        return group

    def _create_control_group(self) -> QtWidgets.QGroupBox:
        '''Create stream control buttons.'''
        group = QtWidgets.QGroupBox('Stream Control')
        layout = QtWidgets.QHBoxLayout(group)

        self.start_btn = QtWidgets.QPushButton('Start')
        self.start_btn.setStyleSheet('''
            QPushButton {
                background-color: #2e7d32;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        ''')

        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet('''
            QPushButton {
                background-color: #c62828;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        ''')

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        return group

    def _create_note_panel(self) -> QtWidgets.QWidget:
        '''Create note panel with developer info and GitHub link.'''
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(4)

        # Developer info
        dev_label = QtWidgets.QLabel('Developed by jkang')
        dev_label.setStyleSheet('color: #666; font-size: 11px;')
        dev_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(dev_label)

        # GitHub link
        github_btn = QtWidgets.QPushButton('GitHub')
        github_btn.setStyleSheet('''
            QPushButton {
                background-color: #24292e;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #444d56;
            }
        ''')
        github_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        github_btn.clicked.connect(self._open_github)
        layout.addWidget(github_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        return widget

    def _open_github(self):
        '''Open GitHub repository in browser.'''
        url = QtCore.QUrl('https://github.com/phoneticscience/rt-waveform-spectrogram')
        QtGui.QDesktopServices.openUrl(url)

    def _connect_signals(self):
        '''Connect widget signals to class signals.'''
        # Real-time parameters
        self.db_min_spin.valueChanged.connect(self._emit_db_range)
        self.db_max_spin.valueChanged.connect(self._emit_db_range)
        self.max_freq_spin.valueChanged.connect(self.max_freq_changed.emit)
        self.colormap_combo.currentTextChanged.connect(self.colormap_changed.emit)
        self.fps_spin.valueChanged.connect(self.fps_target_changed.emit)
        self.wave_gain_spin.valueChanged.connect(self.wave_gain_changed.emit)

        # Stream parameters
        self.apply_btn.clicked.connect(self._emit_stream_settings)

        # Control buttons
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

    def _emit_db_range(self):
        '''Emit dB range if valid.'''
        db_min = self.db_min_spin.value()
        db_max = self.db_max_spin.value()
        if db_min < db_max:
            self.db_range_changed.emit(db_min, db_max)

    def _emit_stream_settings(self):
        '''Emit all stream settings as a dict.'''
        settings = {
            'sr': self.sample_rate_combo.currentData(),
            'seconds': self.duration_spin.value(),
            'blocksize': self.blocksize_combo.currentData(),
            'n_fft': self.nfft_combo.currentData(),
            'hop': self.hop_combo.currentData(),
        }
        self.stream_settings_changed.emit(settings)

    def set_stream_running(self, running: bool):
        '''Update button states based on stream status.'''
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)


class RealtimeAudioViz(QtWidgets.QMainWindow):
    def __init__(self, sr=16000, seconds=2.0, blocksize=256, device=None):
        super().__init__()
        self.setWindowTitle('Realtime Waveform + Spectrogram')

        # Store initial parameters
        self.sr = int(sr)
        self.seconds = float(seconds)
        self.blocksize = int(blocksize)
        self.device = device

        # STFT parameters
        self.n_fft = 1024
        self.hop = 128

        # Initialize audio parameters and buffers
        self._init_audio_params()

        # Setup UI
        self._setup_ui()

        # Connect sidebar signals
        self._connect_sidebar_signals()

        # FPS tracking
        self._last_frame_time = None
        self._fps_ema = 0.0

        # Timer for UI updates
        self.timer = QtCore.QTimer()
        self.timer.setInterval(16)  # ~60 FPS
        self.timer.timeout.connect(self.update_plots)

        self.stream = None

    def _init_audio_params(self):
        '''Initialize audio and STFT parameters.'''
        self.n_samples = int(self.sr * self.seconds)
        self.ring = RingBuffer(self.n_samples)

        # STFT derived parameters
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = self.n_fft // 2 + 1
        self.max_freq = self.sr / 2

        # Fixed dB range (Praat-like: ~50 dB dynamic range)
        self.db_min = -50.0
        self.db_max = 0.0

        # Waveform gain
        self.wave_gain = 4.0

        # Pre-allocate working buffers
        self.audio_buffer = np.empty(self.n_samples, dtype=np.float32)
        self.max_frames = 1 + (self.n_samples - self.n_fft) // self.hop
        self.windowed_frames = np.empty((self.max_frames, self.n_fft), dtype=np.float32)
        self.mag_db_buffer = np.empty((self.freq_bins, self.max_frames), dtype=np.float32)

        # Pre-allocate FFT output buffer (complex)
        self.fft_buffer = np.empty((self.max_frames, self.freq_bins), dtype=np.complex64)
        # Pre-allocate spectrogram display buffer (avoids ascontiguousarray each frame)
        self.spec_display_buffer = np.empty((self.max_frames, self.freq_bins), dtype=np.float32)

        # Waveform downsampling with min/max envelope to prevent aliasing
        self.wave_target_points = 2000
        self.wave_downsample = max(1, self.n_samples // self.wave_target_points)
        # Using min/max envelope doubles the display points
        self.wave_display_len = (self.n_samples // self.wave_downsample) * 2
        self.t_axis = np.linspace(-self.seconds, 0, self.wave_display_len, endpoint=False)
        # Pre-allocate envelope buffer
        self.wave_envelope = np.empty(self.wave_display_len, dtype=np.float32)
        # Pre-allocate waveform display buffer
        self.wave_display_buffer = np.empty(self.wave_display_len, dtype=np.float32)

        # Track if spectrogram rect has been set
        self._spec_rect_set = False

        # FPS update counter (update label every N frames to reduce overhead)
        self._fps_update_counter = 0

    def _init_audio_params_preserve_ring(self):
        '''Reinitialize audio params without recreating ring buffer.'''
        self.n_samples = int(self.sr * self.seconds)

        # STFT derived parameters
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.freq_bins = self.n_fft // 2 + 1
        self.max_freq = self.sr / 2

        # Pre-allocate working buffers
        self.audio_buffer = np.empty(self.n_samples, dtype=np.float32)
        self.max_frames = 1 + (self.n_samples - self.n_fft) // self.hop
        self.windowed_frames = np.empty((self.max_frames, self.n_fft), dtype=np.float32)
        self.mag_db_buffer = np.empty((self.freq_bins, self.max_frames), dtype=np.float32)

        # Pre-allocate FFT output buffer (complex)
        self.fft_buffer = np.empty((self.max_frames, self.freq_bins), dtype=np.complex64)
        # Pre-allocate spectrogram display buffer
        self.spec_display_buffer = np.empty((self.max_frames, self.freq_bins), dtype=np.float32)

        # Waveform downsampling with min/max envelope to prevent aliasing
        self.wave_target_points = 2000
        self.wave_downsample = max(1, self.n_samples // self.wave_target_points)
        self.wave_display_len = (self.n_samples // self.wave_downsample) * 2
        self.t_axis = np.linspace(-self.seconds, 0, self.wave_display_len, endpoint=False)
        # Pre-allocate envelope and display buffers
        self.wave_envelope = np.empty(self.wave_display_len, dtype=np.float32)
        self.wave_display_buffer = np.empty(self.wave_display_len, dtype=np.float32)
        self.wave_envelope = np.empty(self.wave_display_len, dtype=np.float32)

    def _setup_ui(self):
        '''Setup main UI layout.'''
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(central)

        # Sidebar
        self.sidebar = ControlSidebar()
        self.sidebar.setStyleSheet('''
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #0066cc;
            }
        ''')
        main_layout.addWidget(self.sidebar)

        # Plot area
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(8, 8, 8, 8)
        plot_layout.setSpacing(8)

        # FPS display bar
        fps_bar = QtWidgets.QHBoxLayout()
        fps_bar.addStretch()
        self.fps_label = QtWidgets.QLabel('FPS: 0.0')
        self.fps_label.setStyleSheet(
            'color: #008800; font-family: "Courier New", Courier, monospace; font-size: 14px; font-weight: bold;'
        )
        fps_bar.addWidget(self.fps_label)
        plot_layout.addLayout(fps_bar)

        # Configure pyqtgraph
        pg.setConfigOptions(antialias=False, useOpenGL=False)

        # Waveform plot
        self.wave_plot = pg.PlotWidget(title='Waveform')
        self.wave_plot.setXRange(-self.seconds, 0)  # Initialize X-axis range
        self.wave_plot.setYRange(-1.0, 1.0)
        self.wave_plot.setLabel('left', 'Amplitude')
        self.wave_plot.setLabel('bottom', 'Time', units='s')
        self.wave_plot.setMouseEnabled(x=False, y=False)  # Disable scroll-to-zoom
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen(color='#008800', width=1))
        plot_layout.addWidget(self.wave_plot, stretch=1)

        # Spectrogram
        self.spec_plot = pg.PlotWidget(title='Spectrogram')
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        self.spec_plot.setLabel('left', 'Frequency', units='Hz')
        self.spec_plot.setLabel('bottom', 'Time', units='s')
        self.spec_plot.setXRange(-self.seconds, 0)
        # Display frequency range
        self.display_max_freq = min(6000, self.max_freq)
        self.spec_plot.setYRange(0, self.display_max_freq)
        self.spec_plot.setMouseEnabled(x=False, y=False)  # Disable scroll-to-zoom
        # Default grayscale colormap (white=low, black=high like Praat)
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[:, 0] = np.linspace(255, 0, 256)
        lut[:, 1] = np.linspace(255, 0, 256)
        lut[:, 2] = np.linspace(255, 0, 256)
        lut[:, 3] = 255
        self.spec_img.setLookupTable(lut)
        plot_layout.addWidget(self.spec_plot, stretch=2)

        main_layout.addWidget(plot_container, stretch=1)

    def _connect_sidebar_signals(self):
        '''Connect sidebar signals to handler methods.'''
        self.sidebar.db_range_changed.connect(self._on_db_range_changed)
        self.sidebar.max_freq_changed.connect(self._on_max_freq_changed)
        self.sidebar.colormap_changed.connect(self._on_colormap_changed)
        self.sidebar.fps_target_changed.connect(self._on_fps_target_changed)
        self.sidebar.wave_gain_changed.connect(self._on_wave_gain_changed)
        self.sidebar.stream_settings_changed.connect(self._on_stream_settings_changed)
        self.sidebar.start_requested.connect(self.start_stream)
        self.sidebar.stop_requested.connect(self.stop_stream)

    def _on_db_range_changed(self, db_min: float, db_max: float):
        '''Update dB range for spectrogram display.'''
        self.db_min = db_min
        self.db_max = db_max

    def _on_max_freq_changed(self, freq: int):
        '''Update spectrogram frequency display range.'''
        self.display_max_freq = min(freq, self.max_freq)
        self.spec_plot.setYRange(0, self.display_max_freq)

    def _on_colormap_changed(self, name: str):
        '''Change spectrogram colormap immediately.'''
        try:
            if name == 'grayscale':
                # Custom grayscale: white (low) to black (high) like Praat
                lut = np.zeros((256, 4), dtype=np.uint8)
                lut[:, 0] = np.linspace(255, 0, 256)  # R: white to black
                lut[:, 1] = np.linspace(255, 0, 256)  # G
                lut[:, 2] = np.linspace(255, 0, 256)  # B
                lut[:, 3] = 255  # Alpha
            else:
                cmap = pg.colormap.get(name)
                lut = cmap.getLookupTable(0.0, 1.0, 256)
            self.spec_img.setLookupTable(lut)
        except Exception:
            pass

    def _on_fps_target_changed(self, fps: int):
        '''Adjust timer interval based on target FPS.'''
        interval_ms = max(8, int(1000 / fps))
        self.timer.setInterval(interval_ms)

    def _on_wave_gain_changed(self, gain: float):
        '''Update waveform display gain.'''
        self.wave_gain = gain

    def _on_stream_settings_changed(self, settings: dict):
        '''Apply new stream settings, preserving audio data when possible.'''
        was_running = self.stream is not None

        # Check if sample rate changed (requires stream restart)
        sr_changed = self.sr != settings['sr']
        blocksize_changed = self.blocksize != settings['blocksize']

        if was_running and (sr_changed or blocksize_changed):
            self.stop_stream()

        # Check if only duration/FFT params changed (no stream restart needed)
        old_n_samples = self.n_samples

        # Update parameters
        self.sr = settings['sr']
        self.seconds = settings['seconds']
        self.blocksize = settings['blocksize']
        self.n_fft = settings['n_fft']
        self.hop = settings['hop']

        # Calculate new buffer size
        new_n_samples = int(self.sr * self.seconds)

        # Resize ring buffer to preserve existing audio data
        if new_n_samples != old_n_samples:
            self.ring.resize(new_n_samples)

        # Reinitialize other params (but ring buffer already resized)
        self._init_audio_params_preserve_ring()

        # Update waveform plot range
        self.wave_plot.setXRange(-self.seconds, 0)

        # Update spectrogram plot ranges and reset rect flag
        self.spec_plot.setXRange(-self.seconds, 0)
        self.display_max_freq = min(self.display_max_freq, self.max_freq)
        self.spec_plot.setYRange(0, self.display_max_freq)
        self._spec_rect_set = False

        if was_running and (sr_changed or blocksize_changed):
            self.start_stream()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            # Log audio stream issues (overflow/underflow)
            # These are common and not critical, so just track them
            self._audio_status_warning = str(status)
        x = indata[:, 0].astype(np.float32, copy=False)
        self.ring.write(x)

    def _show_error(self, title: str, message: str, detail: str = None):
        '''Show error message dialog.'''
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if detail:
            msg_box.setDetailedText(detail)
        msg_box.exec()

    def _show_warning(self, title: str, message: str, detail: str = None):
        '''Show warning message dialog.'''
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if detail:
            msg_box.setDetailedText(detail)
        msg_box.exec()

    def start_stream(self):
        if self.stream is not None:
            return

        try:
            self.stream = sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype='float32',
                blocksize=self.blocksize,
                callback=self.audio_callback,
                device=self.device,
            )
            self.stream.start()
            self.timer.start()
            self.sidebar.set_stream_running(True)
        except sd.PortAudioError as e:
            self._show_error(
                'Audio Device Error',
                'Failed to open audio input device.',
                f'PortAudio error: {e}\n\n'
                'Possible causes:\n'
                '- No microphone connected\n'
                '- Microphone permission denied\n'
                '- Audio device in use by another application\n'
                '- Unsupported sample rate for this device'
            )
            self.stream = None
        except OSError as e:
            self._show_error(
                'System Audio Error',
                'Operating system audio error occurred.',
                f'Error: {e}\n\n'
                'Please check your system audio settings.'
            )
            self.stream = None
        except Exception as e:
            self._show_error(
                'Unexpected Error',
                'An unexpected error occurred while starting audio.',
                f'Error type: {type(e).__name__}\n'
                f'Details: {e}'
            )
            self.stream = None

    def stop_stream(self):
        self.timer.stop()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
        self.sidebar.set_stream_running(False)

    def compute_stft_mag(self, x: np.ndarray) -> np.ndarray:
        '''Compute STFT magnitude in dB using pre-allocated buffers.'''
        n = x.size
        if n < self.n_fft:
            pad_x = np.zeros(self.n_fft, dtype=np.float32)
            pad_x[:n] = x
            x = pad_x
            n = self.n_fft

        n_frames = 1 + (n - self.n_fft) // self.hop
        if n_frames <= 0:
            return None

        # Strided view (no copy)
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_frames, self.n_fft),
            strides=(x.strides[0] * self.hop, x.strides[0]),
            writeable=False,
        )

        # Apply cached window to pre-allocated buffer
        np.multiply(frames, self.window, out=self.windowed_frames[:n_frames])

        # FFT - use scipy if available (faster), otherwise numpy
        if USE_SCIPY_FFT:
            self.fft_buffer[:n_frames] = scipy_fft.rfft(
                self.windowed_frames[:n_frames], n=self.n_fft, axis=1
            )
        else:
            self.fft_buffer[:n_frames] = np.fft.rfft(
                self.windowed_frames[:n_frames], n=self.n_fft, axis=1
            )

        # Magnitude in-place
        np.abs(self.fft_buffer[:n_frames], out=self.mag_db_buffer[:, :n_frames].T)

        # Convert to dB in pre-allocated buffer (in-place operations)
        mag_view = self.mag_db_buffer[:, :n_frames]
        np.maximum(mag_view, 1e-10, out=mag_view)
        np.log10(mag_view, out=mag_view)
        mag_view *= 20.0

        return mag_view

    def update_plots(self):
        # FPS calculation (update label only every 10 frames to reduce overhead)
        now = time.perf_counter()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self._fps_ema = 0.1 * instant_fps + 0.9 * self._fps_ema
                self._fps_update_counter += 1
                if self._fps_update_counter >= 10:
                    self.fps_label.setText(f'FPS: {self._fps_ema:.1f}')
                    self._fps_update_counter = 0
        self._last_frame_time = now

        # Read into pre-allocated buffer
        x = self.ring.read_latest(self.n_samples, out=self.audio_buffer)

        # Waveform with min/max envelope to prevent aliasing
        if self.wave_downsample > 1:
            # Reshape into chunks and compute min/max per chunk
            n_chunks = self.n_samples // self.wave_downsample
            trimmed = x[:n_chunks * self.wave_downsample]
            chunks = trimmed.reshape(n_chunks, self.wave_downsample)
            # Use pre-allocated envelope buffer with in-place operations
            np.min(chunks, axis=1, out=self.wave_envelope[0::2][:n_chunks])
            np.max(chunks, axis=1, out=self.wave_envelope[1::2][:n_chunks])
            # Apply gain to pre-allocated display buffer
            np.multiply(self.wave_envelope, self.wave_gain, out=self.wave_display_buffer)
            self.wave_curve.setData(self.t_axis, self.wave_display_buffer)
        else:
            np.multiply(x, self.wave_gain, out=self.wave_display_buffer[:len(x)])
            self.wave_curve.setData(self.t_axis[:len(x)], self.wave_display_buffer[:len(x)])

        # Spectrogram
        mag_db = self.compute_stft_mag(x)
        if mag_db is not None:
            n_frames = mag_db.shape[1]
            # Use pre-allocated display buffer (transposed view)
            self.spec_display_buffer[:n_frames, :] = mag_db.T
            self.spec_img.setImage(
                self.spec_display_buffer[:n_frames],
                autoLevels=False,
                levels=(self.db_min, self.db_max)
            )

            if not self._spec_rect_set:
                self.spec_img.setRect(-self.seconds, 0, self.seconds, self.max_freq)
                self._spec_rect_set = True

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


def check_audio_available() -> tuple[bool, str]:
    '''Check if audio input is available.

    Returns:
        (available, message) tuple
    '''
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            return False, 'No audio input devices found.'

        # Try to query default input device
        default_input = sd.query_devices(kind='input')
        if default_input is None:
            return False, 'No default audio input device configured.'

        return True, f"Default input: {default_input['name']}"
    except sd.PortAudioError as e:
        return False, f'PortAudio error: {e}'
    except OSError as e:
        return False, f'System audio error: {e}'
    except Exception as e:
        return False, f'Unexpected error: {e}'


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Check audio availability at startup
    audio_ok, audio_msg = check_audio_available()
    if not audio_ok:
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle('Audio Warning')
        msg_box.setText('Audio input may not be available.')
        msg_box.setInformativeText(
            'The application will start, but audio capture may not work.\n\n'
            f'Details: {audio_msg}'
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Ok |
            QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
        result = msg_box.exec()
        if result == QtWidgets.QMessageBox.StandardButton.Cancel:
            sys.exit(0)

    w = RealtimeAudioViz(sr=16000, seconds=4.0, blocksize=256, device=None)
    w.setMinimumWidth(560)  # At least 2x sidebar width (280px)
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
