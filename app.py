from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from utils.inference import (
    DetectionResult,
    get_available_models,
    infer_image,
    infer_video,
    model_supports_task,
    resolve_model_path,
)
from utils.visualization import detection_table, render_summary_metrics


st.set_page_config(
    page_title="Industrial Product Defect Detection",
    page_icon=":factory:",
    layout="wide",
)


DEFAULT_CLASSES = [
    "scratch",
    "dent",
    "crack",
    "missing_part",
    "paint_defect",
    "contamination",
]

DETECTION_COLUMNS = [
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "area",
    "source_name",
]


def load_image(upload) -> Image.Image:
    image = Image.open(upload)
    return image.convert("RGB")


def parse_class_focus(raw_text: str) -> list[str]:
    values = [item.strip() for item in raw_text.split(",")]
    return [item for item in values if item]


def save_uploaded_weights(upload) -> Path:
    suffix = Path(upload.name).suffix or ".pt"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(upload.getbuffer())
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def show_intro() -> None:
    st.title("Industrial Product Defect Detection")
    st.caption("YOLO-powered visual inspection with an operator-friendly Streamlit dashboard.")

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(
            """
            This dashboard helps teams inspect production imagery and review:

            - detected defect regions
            - confidence scores by class
            - pass/fail quality control status
            - per-run defect trends for uploaded images or videos
            """
        )
    with right:
        st.info(
            "Tip: use a custom-trained YOLO weights file for your factory defect classes. "
            "The default models work as a demo baseline but are not defect-specialized."
        )


def render_sidebar() -> dict:
    st.sidebar.header("Inspection Controls")

    available_models = get_available_models()
    model_choice = st.sidebar.selectbox(
        "Model source",
        options=["Built-in YOLO", "Upload custom weights", "Use local weights path"],
    )

    uploaded_weights = None
    weights_path = None

    if model_choice == "Built-in YOLO":
        weights_path = st.sidebar.selectbox("Checkpoint", options=available_models, index=0)
    elif model_choice == "Upload custom weights":
        uploaded_weights = st.sidebar.file_uploader("Upload .pt weights", type=["pt"])
    else:
        weights_path = st.sidebar.text_input("Local weights path", value="best.pt")

    confidence = st.sidebar.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou = st.sidebar.slider("NMS IoU threshold", min_value=0.10, max_value=0.95, value=0.45, step=0.05)
    max_detections = st.sidebar.slider("Max detections", min_value=1, max_value=300, value=100, step=1)
    fail_threshold = st.sidebar.number_input("Fail if total defects >=", min_value=1, value=1, step=1)
    focus_classes_text = st.sidebar.text_input(
        "Priority defect classes",
        value=", ".join(DEFAULT_CLASSES[:4]),
        help="Comma-separated list used in the quality summary and filtered views.",
    )
    media_type = st.sidebar.radio("Inspection input", options=["Image", "Video"], horizontal=True)

    return {
        "model_choice": model_choice,
        "uploaded_weights": uploaded_weights,
        "weights_path": weights_path,
        "confidence": confidence,
        "iou": iou,
        "max_detections": max_detections,
        "fail_threshold": fail_threshold,
        "focus_classes": parse_class_focus(focus_classes_text),
        "media_type": media_type,
    }


def collect_result_rows(results: Iterable[DetectionResult]) -> pd.DataFrame:
    rows = []
    for item in results:
        for detection in item.detections:
            row = detection.model_dump()
            row["source_name"] = item.source_name
            rows.append(row)
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def render_class_chart(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No detections found for the current run.")
        return

    class_counts = df.groupby("class_name", as_index=False).size().rename(columns={"size": "detections"})
    chart = px.bar(
        class_counts,
        x="class_name",
        y="detections",
        color="class_name",
        title="Detections by defect class",
        labels={"class_name": "Defect class", "detections": "Count"},
    )
    chart.update_layout(showlegend=False, height=380)
    st.plotly_chart(chart, use_container_width=True)


def render_confidence_chart(df: pd.DataFrame) -> None:
    if df.empty:
        return

    box_chart = px.box(
        df,
        x="class_name",
        y="confidence",
        color="class_name",
        title="Confidence distribution by class",
        labels={"class_name": "Defect class", "confidence": "Confidence"},
        points="all",
    )
    box_chart.update_layout(showlegend=False, height=380)
    st.plotly_chart(box_chart, use_container_width=True)


def render_quality_panel(df: pd.DataFrame, fail_threshold: int, focus_classes: list[str]) -> None:
    if df.empty or "class_name" not in df.columns:
        defect_count = 0
        status = "PASS"
    else:
        focus_df = df[df["class_name"].isin(focus_classes)] if focus_classes else df
        defect_count = len(focus_df.index)
        status = "FAIL" if defect_count >= fail_threshold else "PASS"
    tone = "error" if status == "FAIL" else "success"

    getattr(st, tone)(
        f"QC Status: {status} | Priority defects found: {defect_count} | "
        f"Threshold: {fail_threshold}"
    )

    if focus_classes:
        st.caption("Priority classes: " + ", ".join(focus_classes))


def main() -> None:
    show_intro()
    config = render_sidebar()

    uploader_types = ["png", "jpg", "jpeg", "bmp"] if config["media_type"] == "Image" else ["mp4", "avi", "mov", "mkv"]
    uploaded_media = st.file_uploader(
        f"Upload {config['media_type'].lower()} files",
        type=uploader_types,
        accept_multiple_files=True,
    )

    if not uploaded_media:
        st.warning("Upload one or more files to start inspection.")
        return

    weights_path = resolve_model_path(
        model_choice=config["model_choice"],
        selected_path=config["weights_path"],
        uploaded_file=config["uploaded_weights"],
        save_uploaded_fn=save_uploaded_weights,
    )

    if not weights_path:
        st.warning("Select or upload a YOLO weights file to continue.")
        return

    if not model_supports_task(weights_path):
        st.error(
            "The selected weights file could not be loaded. Make sure it is a valid YOLO detection checkpoint."
        )
        return

    if st.button("Run inspection", type="primary", use_container_width=True):
        with st.spinner("Running YOLO inference and building dashboard..."):
            if config["media_type"] == "Image":
                results = [
                    infer_image(
                        image=load_image(file),
                        source_name=file.name,
                        model_path=weights_path,
                        conf=config["confidence"],
                        iou=config["iou"],
                        max_det=config["max_detections"],
                    )
                    for file in uploaded_media
                ]
            else:
                results = [
                    infer_video(
                        upload=file,
                        model_path=weights_path,
                        conf=config["confidence"],
                        iou=config["iou"],
                        max_det=config["max_detections"],
                    )
                    for file in uploaded_media
                ]

        all_detections = collect_result_rows(results)

        render_summary_metrics(results)
        render_quality_panel(all_detections, config["fail_threshold"], config["focus_classes"])

        st.subheader("Visual Review")
        for result in results:
            st.markdown(f"**{result.source_name}**")
            if result.preview_image is not None:
                st.image(result.preview_image, use_container_width=True)
            if result.output_video_path:
                st.video(str(result.output_video_path))

        st.subheader("Analytics")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            render_class_chart(all_detections)
        with chart_right:
            render_confidence_chart(all_detections)

        st.subheader("Detection Log")
        st.dataframe(detection_table(all_detections), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
