# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for rt-waveform-spectrogram

Build with: pyinstaller rt_wav_sgram.spec
"""
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all scipy.fft submodules (needed for FFT operations)
scipy_hiddenimports = collect_submodules('scipy.fft')

# Collect PyQt6 data files
pyqt6_datas = collect_data_files('PyQt6', include_py_files=False)

a = Analysis(
    ['src/rt_wav_sgram.py'],
    pathex=[],
    binaries=[],
    datas=pyqt6_datas,
    hiddenimports=[
        # scipy FFT modules
        'scipy.fft',
        'scipy.fft._pocketfft',
        'scipy.fft._pocketfft.pypocketfft',
        *scipy_hiddenimports,
        # numpy modules
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        # PyQt6 modules
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        # pyqtgraph
        'pyqtgraph',
        'pyqtgraph.graphicsItems',
        'pyqtgraph.graphicsItems.ImageItem',
        'pyqtgraph.graphicsItems.PlotItem',
        # sounddevice
        'sounddevice',
        '_sounddevice_data',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'PIL',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Platform-specific settings
if sys.platform == 'darwin':
    # macOS: Create .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='rt-waveform-spectrogram',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='rt-waveform-spectrogram',
    )
    app = BUNDLE(
        coll,
        name='rt-waveform-spectrogram.app',
        icon=None,  # Add icon path here if available: 'assets/icon.icns'
        bundle_identifier='com.phoneticscience.rt-waveform-spectrogram',
        info_plist={
            'NSMicrophoneUsageDescription': 'This app requires microphone access for real-time audio visualization.',
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '0.1.0',
        },
    )
elif sys.platform == 'win32':
    # Windows: Single executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='rt-waveform-spectrogram',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,  # Add icon path here if available: 'assets/icon.ico'
    )
else:
    # Linux: Folder-based distribution
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='rt-waveform-spectrogram',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='rt-waveform-spectrogram',
    )
