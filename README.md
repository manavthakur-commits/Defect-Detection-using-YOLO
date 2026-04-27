# Industrial Product Defect Detection using YOLO + Streamlit

This project provides a Streamlit dashboard for visual quality inspection of industrial products using YOLO object detection.

## Features

- upload one or more product images or videos
- run YOLO-based defect detection
- review annotated outputs
- inspect defect counts, confidence trends, and QC pass/fail status
- plug in your own trained YOLO `.pt` weights

## Project Structure

- `app.py` - Streamlit dashboard
- `utils/inference.py` - YOLO inference helpers for images and videos
- `utils/visualization.py` - dashboard metrics and detection table formatting
- `requirements.txt` - Python dependencies

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Notes

- For real defect detection, use a custom-trained YOLO model on your manufacturing dataset.
- Built-in YOLO checkpoints are included as demo options and are not trained specifically for industrial defects.
