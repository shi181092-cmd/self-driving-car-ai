# AI Self-Driving Car Simulator

A beginner-friendly educational Streamlit app. It uses a pretrained YOLOv8 Nano model to detect objects in uploaded road images and videos, then applies simple rules to suggest a simulated driving action.

## Run on your computer

1. Install Python 3.10 or 3.11.
2. Download and extract this project folder.
3. Open Terminal (Mac) or Command Prompt/PowerShell (Windows) inside the folder.
4. Create a virtual environment (recommended):

   **Mac/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows**
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```
7. A browser tab should open. If not, copy the local URL shown in Terminal (usually http://localhost:8501).

The first run downloads the YOLO model weights (`yolov8n.pt`), so an internet connection is needed.

## Use an online coding platform

You can import the extracted project into Replit or another Python environment that supports Streamlit. Keep `app.py` and `requirements.txt` in the project root, install the requirements, then run:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Platform setup commands may differ.

## Demo inputs

Use your own road images or short MP4 clips. Public road videos can be used for a classroom demonstration if you have permission to use them.

## Safety and limitations

This is a classroom simulation only. It is not a certified autonomous-driving system and must never be connected to or used to control a real car. Its simple rules and traffic-light colour estimate are not reliable enough for real driving.
