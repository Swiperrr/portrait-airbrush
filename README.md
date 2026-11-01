# Portrait Airbrush

Upload a portrait and get three retouched versions (subtle, medium and strong), plus a side-by-side comparison. The skin gets a smooth, magazine-style finish that still looks like a real photograph.

**Try it online:** https://huggingface.co/spaces/swiperrr/portrait-airbrush

It uses classical image processing only. There is no generative AI, no inpainting and no face reshaping. MediaPipe is used only to find the face and to tell skin from hair and clothes.

## Run it

You need Python 3.11.

```bash
git clone https://github.com/Swiperrr/portrait-airbrush.git
cd portrait-airbrush
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Open http://127.0.0.1:7860, upload a photo (JPG, PNG or HEIC) and press **Airbrush**. Add `--share` to get a temporary public link you can send to others.

The two MediaPipe model files (about 20 MB) download automatically on first run.

### Command line

```bash
.venv/bin/python retouch.py photo.jpg another.heic --outdir output
```

For each photo this writes `_subtle`, `_medium`, `_strong` and `_airbrushed` (the same as medium) at full resolution with the original colour profile (JPEG quality 96), plus `_comparison.jpg`.

## How it works

1. **Skin mask.** It finds the face landmarks and segments face skin, body skin, hair and clothes. The mask is refined at the face for sharp edges and snapped to real edges with a guided filter. It leaves out the eyes, brows, lips, nostrils, hair and jewellery, then feathers the edges in proportion to the face size.
2. **Blemish cleanup.** Small dark or red spots are healed with a median blend. Moles and clusters of freckles are left alone.
3. **Frequency separation.** The tone layer is evened with an edge-preserving guided filter. The texture layer keeps about half of the pore detail. Strong contours such as the jaw, eyelids and nostrils are protected, so no halos form.
4. **Dodge and burn.** Under-eye shadows are lifted toward the cheek tone and excess redness is reduced slightly. The natural shading stays.
5. **Finish.** The retouched skin is blended over the original at about 80%. Then come light clarity on the eyes and hair, a soft glow on highlights, and fine film grain over the whole frame.

The strength settings are at the top of `retouch.py`.

## Bonus: film polish

`polish.py` gives any photo (no face needed) a grainy, film-style look: faded blacks, gentle contrast, warm highlights, glow, vignette and grain.

```bash
.venv/bin/python polish.py photo.heic
```

## Licence

MIT. MediaPipe and its models are Apache 2.0 (Google).
