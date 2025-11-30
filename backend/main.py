import random
import torch
import cv2
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, csv, sys, base64
from io import BytesIO
import pillow_heif             # HEIC/HEIF support for Pillow
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision.transforms import ToPILImage

# ── Make model_src importable ───────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from model_src.CNN import NeuralNetwork

# ── Register the HEIC/HEIF opener with Pillow ───────────────────────────────
pillow_heif.register_heif_opener()

# ── Instantiate FastAPI ─────────────────────────────────────────────────────
app = FastAPI(
    title="EcoPulse API",
    description="Serve training metrics CSVs + inference to your frontend",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in prod, restrict to your real domain
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Static‐files for metrics CSVs ────────────────────────────────────────────
project_dir = os.path.join(os.path.dirname(__file__), "data", "metrics_csv")
app.mount("/metrics", StaticFiles(directory=project_dir), name="metrics")

to_pil = ToPILImage()

# ── Load pretrained model for inference ──────────────────────────────────────
project = "test_2"
load_checkpoint_path = os.path.join(
    ROOT, "model_src", "checkpoints", project, "best_model.pth.tar"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(DEVICE)
checkpoint = torch.load(load_checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ── Prepare MNIST test dataset for random inference ───────────────────────────
TEST_ROOT = os.path.join(ROOT, "model_src", "data")
test_dataset = datasets.MNIST(
    root=TEST_ROOT,
    train=False,
    download=False,
    transform=transforms.ToTensor(),
)

# ── Helpers for metrics listing ──────────────────────────────────────────────
def list_projects() -> list[str]:
    try:
        entries = os.listdir(project_dir)
    except FileNotFoundError:
        return []
    return [n for n in entries if os.path.isdir(os.path.join(project_dir, n))]

def list_tags(proj: str) -> list[str]:
    d = os.path.join(project_dir, proj)
    if not os.path.isdir(d):
        raise HTTPException(404, detail=f"Unknown project: {proj}")
    return [os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".csv")]

# ── Metrics JSON API ────────────────────────────────────────────────────────
@app.get("/api/projects", summary="List all project names")
def api_list_projects():
    return {"projects": list_projects()}

@app.get("/api/{project}/tags", summary="List all metric tags in a project")
def api_list_tags(project: str):
    return {"project": project, "tags": list_tags(project)}

@app.get("/api/{project}/{tag}", summary="Fetch a single metric as JSON")
def api_fetch_metric(project: str, tag: str):
    csv_path = os.path.join(project_dir, project, f"{tag}.csv")
    if not os.path.isfile(csv_path):
        raise HTTPException(404, detail="Metric not found")
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)  # expects “step,value” columns
        for r in reader:
            rows.append({"step": int(r["step"]), "value": float(r["value"])})
    return {"project": project, "tag": tag, "data": rows}

# ── Random inference endpoint ────────────────────────────────────────────────
@app.get("/api/random-inference", summary="Run random MNIST inference")
def random_inference():
    idx = random.randrange(len(test_dataset))
    img_tensor, actual = test_dataset[idx]

    # Convert tensor → PNG data URL
    pil = to_pil(img_tensor)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Run model
    with torch.no_grad():
        logits = model(img_tensor.to(DEVICE).unsqueeze(0))
        pred = int(logits.argmax(dim=1)[0])

    return {"actual": actual, "predicted": pred, "image": data_url}

# ── Custom inference via POST ────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    image: str  # data URL, e.g. "data:image/heic;base64,…" or "data:image/png;base64,…"

class InferenceResponse(BaseModel):
    predicted: int
    image: str   # the processed 28×28 PNG returned as data URL

@app.post(
    "/api/custom-inference",
    response_model=InferenceResponse,
    summary="Run inference on an uploaded image",
)
def custom_inference(req: InferenceRequest):
    # 1) Strip off the "data:image/...;base64," prefix
    try:
        header, b64data = req.image.split(",", 1)
    except ValueError:
        raise HTTPException(400, detail="Invalid data URL")

    # 2) Base64 → raw bytes
    raw = base64.b64decode(b64data)

    # 3) Decode to a grayscale NumPy array:
    if "heic" in header or "heif" in header:
        # PIL knows how to open HEIC now that we registered the opener
        pil_img = Image.open(BytesIO(raw)).convert("L")
        gray = np.array(pil_img)
    else:
        arr = np.frombuffer(raw, np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise HTTPException(400, detail="Could not decode image")

        # 4) Binarize & clean: white digit on black
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # remove small speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 5) Find the digit’s bounding box on the CLEAN mask
    coords = cv2.findNonZero(clean)
    if coords is None:
        raise HTTPException(400, detail="No digit found")
    x, y, w, h = cv2.boundingRect(coords)

    # 6) Crop the mask itself
    digit_mask = clean[y:y+h, x:x+w]

    # 7) scale so the *largest* side fills 28px **
    # desired padding
    pad = 2
    # maximum box size for the digit itself
    box = 28 - 2*pad    # = 24px

    max_dim = max(w, h)
    scale    = box / max_dim
    new_w    = int(w * scale)
    new_h    = int(h * scale)

    digit_resized = cv2.resize(
        digit_mask,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    # 8) Center in a 28×28 BLACK canvas
    canvas = np.zeros((28,28), dtype=np.uint8)
    x_off = (28 - new_w)//2
    y_off = (28 - new_h)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized

    # ---- from here on, 'canvas' is your final 28×28 mask ----
    final = canvas  # white digit (255) on black (0)

    # 7) Normalize → tensor & run model
    tensor = torch.from_numpy(final.astype(np.float32) / 255.0) \
                   .unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        pred = int(logits.argmax(dim=1)[0])

    # 8) Re-encode 28×28 PNG for the frontend
    _, png = cv2.imencode(".png", final)
    data_url = (
      "data:image/png;base64," +
      base64.b64encode(png.tobytes()).decode("ascii")
    )

    return InferenceResponse(predicted=pred, image=data_url)
