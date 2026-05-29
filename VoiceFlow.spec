"""
VoiceFlow - PyInstaller spec file for building Windows .exe

Build command (on Windows or via GitHub Actions):
    pyinstaller VoiceFlow.spec --clean

Output: dist/VoiceFlow.exe (~50MB without model, model downloads on first run)
"""

block_cipher = None

a = Analysis(
    ['voiceflow/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include tray icon if available
        # ('icons/mic.ico', 'icons'),
    ],
    hiddenimports=[
        'faster_whisper',
        'faster_whisper.backends',
        'openai',
        'sounddevice',
        'numpy',
        'pynput',
        'pynput.keyboard',
        'pynput.keyboard._win32',
        'pynput.mouse',
        'pynput.mouse._win32',
        'pystray',
        'pystray._win32',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'pkg_resources',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages to reduce .exe size
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'scipy',
        'pandas',
        'IPython',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove unnecessary binaries
a.binaries = [b for b in a.binaries if not b[0].startswith((
    'libcrypto', 'libssl',  # Not needed for basic HTTPS
))]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VoiceFlow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app (no terminal window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icons/mic.ico',  # Uncomment when icon is available
    version='version_info.txt',  # Optional: version info file
)
