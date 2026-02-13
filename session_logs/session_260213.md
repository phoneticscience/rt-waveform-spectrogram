# Session Log: 2026-02-13

## Initial Issues

1. **App slows down and crashes over time** - Memory/buffer issues causing performance degradation
2. **Spectrogram scrolls wrong direction** - Was updating top-to-bottom instead of left-to-right

## Root Causes

### Performance Issues
- Window function (`np.hanning()`) recreated every frame (30x/sec)
- `autoLevels=True` scanned entire image every frame
- Ring buffer created unnecessary array copies
- Too many waveform points rendered (32,000)

### Orientation Issues
- Array shape misinterpreted by pyqtgraph ImageItem
- Frequency axis was inverted

## Changes Made (Chronological)

### 1. Performance & Orientation Fixes
- Cached Hanning window in `__init__` (created once)
- Pre-allocated buffers: audio, windowed frames, magnitude
- Fixed dB range `[-80, 0]` instead of `autoLevels=True`
- Optimized ring buffer to accept pre-allocated output
- Downsampled waveform to ~2,000 points
- Disabled antialiasing and OpenGL (caused flickering)
- Time flows left-to-right, low frequencies at bottom

### 2. UI Sidebar with Controls
- Added `ControlSidebar` class with real-time and stream parameters
- Real-time: dB range, colormap, FPS target, wave gain, max frequency
- Stream params (require Apply): sample rate, duration, block size, FFT size, hop size
- Start/Stop buttons for stream control
- FPS tracker display

### 3. Light Mode & Scroll Zoom Fix
- Removed dark mode theme, switched to default light mode
- Disabled scroll-to-zoom on both plots (`setMouseEnabled(x=False, y=False)`)

### 4. Waveform Fixes
- Fixed waveform X-axis not updating after Apply Changes
- Fixed waveform flickering using min/max envelope method
- Added Wave Gain parameter (1x-100x, default 4x)

### 5. Colormap Improvements
- Fixed colormap not updating in real-time (use `setLookupTable()`)
- Added custom grayscale colormap (white=silence, black=loud) like Praat
- Set grayscale as default colormap

### 6. Praat-like Spectrogram Settings
- Narrowband FFT (n_fft=2048) for better formant visualization
- Smaller hop size (64) for smooth display
- dB range: -50 to 0 (50 dB dynamic range like Praat)
- Default max frequency: 5000 Hz (speech range)
- Max Freq parameter bounded by Nyquist frequency

### 7. Buffer Preservation on Duration Change
- Added `RingBuffer.resize()` method to preserve audio data
- Duration changes no longer clear the display abruptly

### 8. Default Parameter Updates
- Default duration: 4 seconds
- Default wave gain: 4x
- Minimum window width: 560px (2x sidebar width)

### 9. Developer Note Panel
- Added note panel at bottom of sidebar
- "Developed by jkang" label
- GitHub button linking to repository

### 10. Wideband Spectrogram (Praat-like)
- Changed default FFT from 2048 to 1024 for wideband display
- Hop size 128 for smooth time resolution

### 11. Unit Test Suite
- Created `tests/test_rt_wav_sgram.py` with 38 tests
- Test classes: RingBuffer, STFT, ControlSidebar, RealtimeAudioViz, Performance, AudioCheck, ErrorHandling
- Added pytest configuration in `pyproject.toml`

### 12. Error/Warning Dialogs
- Added `check_audio_available()` function for startup audio check
- Added `_show_error()` and `_show_warning()` methods
- Startup warning dialog if no audio device available
- Detailed error messages for audio stream failures

### 13. Updated Default Values
- Max Freq: 6000 Hz
- Target FPS: 60
- Block Size: 256
- FFT Size: 1024
- Hop Size: 128

### 14. Performance Optimization
- Added scipy.fft for faster FFT (with numpy fallback)
- Pre-allocated buffers: fft_buffer, spec_display_buffer, wave_display_buffer
- In-place numpy operations to reduce memory allocation
- FPS label updates every 10 frames instead of every frame

### 15. Cross-Platform Build Setup
- Created `rt_wav_sgram.spec` - PyInstaller configuration
- Created `scripts/build.py` - Cross-platform build script
- Created `.github/workflows/build-release.yml` - GitHub Actions CI/CD
- Builds triggered on version tags (v*.*.*)
- Outputs: macOS .app, Windows .exe, Linux folder

### 16. README Update
- Added download section with direct links to releases
- Platform-specific installation instructions
- Run from source and build from source sections

### 17. Initial Git Commit & Release
- Initial commit pushed to GitHub
- Tag v0.1.0 created and pushed
- GitHub Actions workflow triggered for automated builds

## Files Modified/Created
- `src/rt_wav_sgram.py` - Main application with all features
- `tests/test_rt_wav_sgram.py` - 38 unit tests
- `tests/__init__.py` - Test package init
- `pyproject.toml` - Dependencies and pytest config
- `rt_wav_sgram.spec` - PyInstaller spec file
- `scripts/build.py` - Build automation script
- `.github/workflows/build-release.yml` - CI/CD workflow
- `README.md` - Updated with download/install instructions

## Result
App runs smoothly with Praat-like spectrogram visualization, adjustable parameters, stable performance, comprehensive tests, and cross-platform automated builds.
