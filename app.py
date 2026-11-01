#!/usr/bin/env python3
"""
Web page for the portrait airbrush: upload a photo, get subtle / medium / strong
versions and a side-by-side comparison.

    .venv/bin/python app.py            # opens on http://127.0.0.1:7860
    .venv/bin/python app.py --share    # also prints a temporary public link
"""
import argparse
import os
import re
import tempfile

import gradio as gr
from PIL import Image

import retouch

PREVIEW_SIZE = 1600  # longest side of the on-page previews; downloads stay full size


def _preview(path, outdir):
    im = Image.open(path)
    im.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
    p = os.path.join(outdir, "preview_" + os.path.basename(path))
    im.save(p, "JPEG", quality=90)
    return p


def run(upload, progress=gr.Progress()):
    if upload is None:
        raise gr.Error("Upload a photo first.")
    path = upload if isinstance(upload, str) else upload.name
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(path))[0]) or "photo"
    outdir = tempfile.mkdtemp(prefix="airbrush_")
    # Copy under a clean name so output files are named after the photo.
    src = os.path.join(outdir, base + os.path.splitext(path)[1].lower())
    with open(path, "rb") as f, open(src, "wb") as g:
        g.write(f.read())

    progress(0.1, desc="Finding the face and skin")
    try:
        files = retouch.process(src, outdir)
    except retouch.NoFaceError:
        raise gr.Error("No face found. Use a photo where the face is clearly visible.")
    except Exception as e:  # unreadable file and similar
        raise gr.Error(f"Could not process this photo: {e}")

    progress(0.95, desc="Preparing previews")
    gallery = [(_preview(files[k], outdir), k.capitalize()) for k in ("subtle", "medium", "strong")]
    downloads = [files[k] for k in ("subtle", "medium", "strong", "comparison")]
    return files["comparison"], gallery, downloads


with gr.Blocks(title="Portrait Airbrush") as demo:
    gr.Markdown(
        "# Portrait Airbrush\n"
        "Upload a portrait. You get three versions (subtle, medium, strong) with a "
        "smooth, magazine-style skin finish that still looks like a real photo, plus a comparison.\n\n"
        "Classical image processing only: no AI generation and no reshaping. "
        "Full resolution and the original colour profile are kept."
    )
    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Your photo (JPG, PNG or HEIC)",
                             file_types=["image", ".heic", ".heif"], type="filepath")
            go = gr.Button("Airbrush", variant="primary")
            downloads = gr.File(label="Full-size downloads", file_count="multiple")
        with gr.Column(scale=2):
            comp = gr.Image(label="Comparison: original, subtle, medium, strong (bottom row at 100%)",
                            type="filepath", interactive=False)
            gallery = gr.Gallery(label="Versions", columns=3, height="auto", object_fit="contain")
    go.click(run, inputs=upload, outputs=[comp, gallery, downloads])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="also create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.queue(default_concurrency_limit=1).launch(share=a.share, server_port=a.port)
