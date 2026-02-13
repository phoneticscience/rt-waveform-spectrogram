#!/usr/bin/env python3
"""
Cross-platform build script for rt-waveform-spectrogram.

Usage:
    python scripts/build.py          # Build for current platform
    python scripts/build.py --clean  # Clean build artifacts first
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def get_platform_name() -> str:
    """Get a normalized platform name."""
    system = platform.system().lower()
    if system == 'darwin':
        return 'macos'
    elif system == 'windows':
        return 'windows'
    else:
        return 'linux'


def clean_build_artifacts(project_root: Path):
    """Remove build artifacts from previous builds."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['*.pyc', '*.pyo', '*.spec.bak']

    print("Cleaning build artifacts...")

    for dir_name in dirs_to_clean:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  Removing {dir_path}")
            shutil.rmtree(dir_path)

    # Clean __pycache__ in subdirectories
    for pycache in project_root.rglob('__pycache__'):
        print(f"  Removing {pycache}")
        shutil.rmtree(pycache)


def run_pyinstaller(project_root: Path) -> bool:
    """Run PyInstaller with the spec file."""
    spec_file = project_root / 'rt_wav_sgram.spec'

    if not spec_file.exists():
        print(f"Error: Spec file not found: {spec_file}")
        return False

    print(f"Building with PyInstaller...")
    print(f"  Spec file: {spec_file}")
    print(f"  Platform: {get_platform_name()}")

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--clean',
        '--noconfirm',
        str(spec_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"PyInstaller failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: PyInstaller not found. Install it with: pip install pyinstaller")
        return False


def create_archive(project_root: Path) -> Path | None:
    """Create a distributable archive of the built application."""
    dist_dir = project_root / 'dist'
    platform_name = get_platform_name()
    app_name = 'rt-waveform-spectrogram'

    if not dist_dir.exists():
        print("Error: dist directory not found")
        return None

    archive_name = f"{app_name}-{platform_name}"

    if platform_name == 'macos':
        # For macOS, the .app bundle is in dist/
        app_bundle = dist_dir / f"{app_name}.app"
        if app_bundle.exists():
            archive_path = dist_dir / f"{archive_name}.zip"
            print(f"Creating archive: {archive_path}")
            shutil.make_archive(
                str(dist_dir / archive_name),
                'zip',
                dist_dir,
                f"{app_name}.app"
            )
            return archive_path
        # Fall back to folder if .app doesn't exist
        app_folder = dist_dir / app_name
        if app_folder.exists():
            archive_path = dist_dir / f"{archive_name}.zip"
            print(f"Creating archive: {archive_path}")
            shutil.make_archive(
                str(dist_dir / archive_name),
                'zip',
                dist_dir,
                app_name
            )
            return archive_path

    elif platform_name == 'windows':
        # Windows: Single .exe file
        exe_file = dist_dir / f"{app_name}.exe"
        if exe_file.exists():
            archive_path = dist_dir / f"{archive_name}.zip"
            print(f"Creating archive: {archive_path}")
            shutil.make_archive(
                str(dist_dir / archive_name),
                'zip',
                dist_dir,
                f"{app_name}.exe"
            )
            return archive_path

    else:
        # Linux: Folder distribution
        app_folder = dist_dir / app_name
        if app_folder.exists():
            archive_path = dist_dir / f"{archive_name}.tar.gz"
            print(f"Creating archive: {archive_path}")
            shutil.make_archive(
                str(dist_dir / archive_name),
                'gztar',
                dist_dir,
                app_name
            )
            return archive_path

    print("Warning: Could not find built application to archive")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Build rt-waveform-spectrogram for the current platform'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean build artifacts before building'
    )
    parser.add_argument(
        '--no-archive',
        action='store_true',
        help='Skip creating distributable archive'
    )
    args = parser.parse_args()

    project_root = get_project_root()
    print(f"Project root: {project_root}")
    print(f"Platform: {get_platform_name()}")
    print()

    if args.clean:
        clean_build_artifacts(project_root)
        print()

    # Run PyInstaller
    if not run_pyinstaller(project_root):
        sys.exit(1)

    print()
    print("Build completed successfully!")

    # Create archive
    if not args.no_archive:
        print()
        archive_path = create_archive(project_root)
        if archive_path:
            print(f"Archive created: {archive_path}")
            print(f"Archive size: {archive_path.stat().st_size / 1024 / 1024:.1f} MB")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
