🏭 Industrial Product Defect Detection Dashboard
A robust, structured Streamlit application designed for visual quality inspection in industrial manufacturing pipelines. This tool bridges the gap between raw computer vision models and operator-friendly quality control by leveraging YOLO object detection to identify, classify, and track defects in production imagery and video.

Built with a focus on clean architecture and reproducible analytics, the dashboard allows engineers to move beyond treating AI as a black box by providing transparent, fine-grained control over inference parameters and comprehensive visual logs.

✨ Key Features
Multi-Modal Inspection Pipeline: Process and analyze both images (PNG, JPG, BMP) and video feeds (MP4, AVI, MOV) for automated defect detection.

Bring-Your-Own-Weights (BYOW): Seamlessly plug in custom-trained YOLO .pt weights specifically tailored to your factory's unique defect classes (e.g., scratches, dents, missing parts).

Granular Inference Control: Take the engine apart and tune model mechanics directly from the UI by adjusting confidence thresholds, NMS IoU thresholds, and maximum detection limits.

Automated Quality Control (QC): Configure pass/fail thresholds based on priority defect counts to instantly evaluate production quality.

Rich Analytics & Documentation: Generates interactive Plotly charts tracking defect distributions and confidence scores, alongside tabular data logging for rigorous, academic-grade review.

🛠️ Architecture & Tech Stack
This project prioritizes modularity, separating inference logic and visualization utilities to ensure a highly maintainable codebase.

Frontend & Analytics: Streamlit (>=1.44.0), Plotly (>=5.24.0), Pandas (>=2.2.0)

Computer Vision Engine: Ultralytics YOLO (>=8.3.0), OpenCV (>=4.10.0), Pillow (>=10.0.0)

Data Validation: Pydantic (>=2.8.0) for rigorous validation of detection outputs
