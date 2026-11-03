import os

import gradio as gr
from ultralytics import YOLO

MODEL_FILES = {
    "Detailed model (7 classes)": os.environ.get("MODEL_PATH", "best.pt"),
    "Merged model (4 classes, higher precision)": "best_merged.pt",
}
# keep only the models whose weights file exists
MODEL_FILES = {k: v for k, v in MODEL_FILES.items() if os.path.exists(v)}
_loaded = {}


def get_model(choice):
    if choice not in _loaded:
        _loaded[choice] = YOLO(MODEL_FILES[choice])
    return _loaded[choice]


def detect(image, conf, choice):
    """image: numpy RGB array from Gradio."""
    if image is None:
        return None, "Upload an image first."
    model = get_model(choice)
    result = model.predict(image, conf=conf, iou=0.4, imgsz=640, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    counts = {}
    for c in result.boxes.cls.tolist():
        name = model.names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return annotated, "No road damage detected."
    summary = "\n".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    return annotated, summary


theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="rose",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
).set(
    body_background_fill="#fdf3f6",
    body_background_fill_dark="#fdf3f6",
    body_text_color="#4a2c3a",
    body_text_color_dark="#4a2c3a",
    body_text_color_subdued="#7a5668",
    body_text_color_subdued_dark="#7a5668",
    block_background_fill="#fffafc",
    block_background_fill_dark="#fffafc",
    block_border_color="#f3d9e3",
    block_border_color_dark="#f3d9e3",
    block_label_background_fill="#f9e4ec",
    block_label_background_fill_dark="#f9e4ec",
    block_label_text_color="#6b3a50",
    block_label_text_color_dark="#6b3a50",
    block_title_text_color="#6b3a50",
    block_title_text_color_dark="#6b3a50",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    button_primary_background_fill="#c9668a",
    button_primary_background_fill_dark="#c9668a",
    button_primary_background_fill_hover="#b4547a",
    button_primary_background_fill_hover_dark="#b4547a",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
)

css = """
.gradio-container {background: linear-gradient(180deg, #fdf3f6 0%, #fbeaf0 100%) !important;}
h1 {color: #8a3558 !important; text-align: center;}
.prose, .prose p, .prose em {color: #4a2c3a !important;}
"""

# Gradio 6 takes theme/css in launch(); older versions take them in Blocks()
NEW_GRADIO = int(gr.__version__.split(".")[0]) >= 6
style = dict(theme=theme, css=css)

with gr.Blocks(title="Road Damage Detection", **({} if NEW_GRADIO else style)) as demo:
    gr.Markdown("# 🛣️ Road Damage Detection\nUpload a road image and the model will draw boxes around the damage.\n\n"
                "**Detailed model:** alligator, block, crack, longitudinal, transverse, pothole (+ edge).  \n"
                "**Merged model:** alligator, block, pothole, and *linear_crack* (crack + longitudinal + transverse + edge). "
                "It has higher precision and validation score, but finds slightly fewer damages (lower recall).")
    choice = gr.Radio(list(MODEL_FILES), value=list(MODEL_FILES)[0], label="Model")
    conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
    with gr.Row():
        img_in = gr.Image(type="numpy", label="Input image")
        img_out = gr.Image(label="Detections")
    summary = gr.Textbox(label="Detected damage", lines=6)
    gr.Button("Detect", variant="primary").click(detect, [img_in, conf, choice], [img_out, summary])
    gr.Markdown("*For educational purposes only.*")

if __name__ == "__main__":
    demo.launch(**(style if NEW_GRADIO else {}))
