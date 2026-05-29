"""Setup script for VoiceFlow."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="voiceflow",
    version="0.1.0",
    author="VoiceFlow Contributors",
    description="Free, open-source, system-wide voice dictation app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/voiceflow/voiceflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Utilities",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-mock>=3.0.0", "flake8>=6.0.0"],
        "gpu": ["torch>=2.0.0", "torchaudio>=2.0.0"],
        "all": [
            "pytest>=7.0.0", "pytest-mock>=3.0.0", "flake8>=6.0.0",
            "torch>=2.0.0", "torchaudio>=2.0.0",
            "pyperclip>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "voiceflow=voiceflow.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
