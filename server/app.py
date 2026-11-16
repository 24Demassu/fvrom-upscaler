"""Web app for the upscaler (Gradio). Runs on any machine with PyTorch: CPU, CUDA or ROCm, or on a hosted Gradio platform."""
import os
import tempfile

import gradio as gr
from PIL import Image

import core

try:  # on a ZeroGPU Space the GPU is attached only while the decorated function runs
    import spaces

    gpu = spaces.GPU(duration=int(os.environ.get("GPU_SECONDS", "60")))
except ImportError:  # any other machine: nothing to do
    def gpu(fn):
        return fn

MAX_INPUT_PIXELS = int(os.environ.get("MAX_INPUT_PIXELS", "2100000"))  # ~1920x1080
PREVIEW_SIDE = 2048  # the before/after slider shows a smaller copy; the download is full size

core.preload()  # load weights onto the CPU at start-up, not on the first visitor's quota

CHOICES = [(label, key) for key, (label, _) in core.MODELS.items()]


@gpu
def _upscale(img, model_id, scale):
    return core.upscale(img, model_id, scale)


def run(img, model_id, scale, fmt):
    if img is None:
        raise gr.Error("Drop an image first.")
    w, h = img.size
    if w * h > MAX_INPUT_PIXELS:
        raise gr.Error(
            f"This image is {w}x{h}. The limit is {MAX_INPUT_PIXELS / 1e6:.1f} megapixels "
            "(about 1920x1080). Resize it a bit and try again."
        )
    scale = int(scale)
    out = _upscale(img, model_id, scale)

    ext = "png" if fmt == "PNG" else "jpg"
    path = os.path.join(tempfile.mkdtemp(), f"upscaled_x{scale}.{ext}")
    if fmt == "PNG":
        out.save(path, optimize=False)
    else:
        out.convert("RGB").save(path, quality=95, subsampling=0)

    preview = out.copy()  # the slider shows a smaller copy; the download stays full size
    preview.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
    original = img.convert(out.mode).resize(preview.size, Image.LANCZOS)
    return (original, preview), path


def reset():
    return None, None


with gr.Blocks(title="Image Upscaler", theme=gr.themes.Soft(), delete_cache=(3600, 3600)) as demo:
    gr.Markdown(
        "# Image Upscaler\n"
        "Drop an image to make it 4x sharper. Free, no sign-in. "
        "Drag the slider to compare."
    )
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(label="Your image", type="pil", sources=["upload", "clipboard"], height=320)
            model = gr.Radio(CHOICES, value=core.DEFAULT_MODEL, label="Style")
            with gr.Row():
                scale = gr.Radio(["2", "4"], value="4", label="Scale (x)")
                fmt = gr.Radio(["PNG", "JPG"], value="PNG", label="Download as")
            btn = gr.Button("Upscale again with these settings", variant="secondary")
        with gr.Column(scale=2):
            result = gr.ImageSlider(label="Before / after", type="pil", show_download_button=False)
            file = gr.File(label="Download full-resolution result", interactive=False)
    gr.Markdown(
        f"Up to {MAX_INPUT_PIXELS / 1e6:.1f} megapixels per image. Uploads start automatically. "
        "Files are deleted automatically after about an hour. "
        "Powered by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)."
    )

    inputs, outputs = [img, model, scale, fmt], [result, file]
    img.upload(run, inputs, outputs, api_name="upscale")
    btn.click(run, inputs, outputs, api_name=False)
    img.clear(reset, None, outputs, api_name=False)

demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch()
