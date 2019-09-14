# Drone_Tracker_Python
Drone tracking algorithm implementation in python based on older algorithm implemented in Matlab (see matlab_drone_tracker_project_report.docx)
<br>Full project description for the new algorithm can be found in the project report file: python_drone_tracker_project_report.pdf


<b>Software Installation for windows 64 bit:</b> (Detailed installation instructions can be found in setup_instructions.pdf)
* Anaconda for Python 3.5 -> https://repo.continuum.io/archive/ -> Anaconda3-4.2.0-Windows-x86_64.exe
  * Add Anaconda3 to Environment Variables
* JetBrains PyCharm 2019.1.1 -> https://www.jetbrains.com/pycharm/download/#section=windows
  * Configure Anaconda as your project's python interpreter
* Python packages
  * opencv-python (3.4.5.20)
  * joblib (0.13.2)
  * numpy (1.11.3)
  * scipy (1.2.1)
  * scikit-image (0.13.1)
  * sklearn (0.0)
  * scikit-learn (0.20.3)
  * matplotlib (1.5.1)
  * Pillow (2.9.0)
* Caffe -> https://github.com/BVLC/caffe/tree/windows
  * edit scripts\build_win.cmd:
    * WITH_NINJA = 0
    * PYTHON_VERSION = 3
    * CONDA_ROOT = <your conda root, e.g C:\Program Files\Anaconda3>
* Visual Studio 2015 -> https://visualstudio.microsoft.com/vs/older-downloads/
  * Note that the compiler is not installed by default, choose C++ in Languages during installation
* CMake -> https://cmake.org/download/ -> cmake-3.14.5-win64-x64.msi
* Git -> https://git-scm.com/download/win
* Additional files
  * bvlc_alexnet - a folder with the neural network configuration and Caffe model <b> TODO: link to Drive </b>
  * Drone movies (GOPR0010.MP4, GOPR0014.mp4) - <b> TODO: link to Drive </b>
* Edit paths in Python code
  * Main_Drine_Tracking.py - net Caffe model path (lines 21-22)
  * video_info.py - input videos path
