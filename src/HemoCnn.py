import tkinter as tk
from tkinter import messagebox
import os, sys, json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk

def ask_open_file():
    root = tk.Tk()
    root.withdraw()
    filename = root.tk.call('tk_getOpenFile')
    root.destroy()
    return filename

tk._default_root = None

APP_NAME = "HemoCnn"
VERSION = "1.1.1"

CLASSES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Immature Granulocyte (IG)",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet"
]

def get_shared_dir():
    appdata = os.getenv("APPDATA")
    shared = os.path.join(appdata, "HemoCnn")
    os.makedirs(shared, exist_ok=True)
    return shared

def get_assets_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "assets")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

SETTINGS_PATH = os.path.join(get_shared_dir(), "settings.json")

def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_settings(data):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=4)

settings = load_settings()

model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path):
    global model
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)

    return CLASSES[pred.item()], conf.item() * 100

def test_cell():
    model_path = settings.get("model_path")

    if not model_path or not os.path.exists(model_path):
        messagebox.showerror("Model not set",
            "Please set the path to blood_cells_classifier.pth in Settings.")
        return

    img_path = ask_open_file()
    if not img_path:
        return
    global model
    if model is None:
        load_model(model_path)

    label, confidence = predict_image(img_path)

    messagebox.showinfo(
        "Prediction Result",
        f"🧬 Predicted Cell Type:\n\n{label}\n\n"
        f"Confidence: {confidence:.2f}%"
    )

def show_settings():
    win = tk.Toplevel(root)
    win.title("🛠 Settings")
    win.geometry("300x180")
    win.transient(root)
    win.grab_set()
    tk.Label(win, text="Path to model (.pth):").pack(anchor="w", padx=10, pady=(15, 5))
    model_var = tk.StringVar(value=settings.get("model_path", ""))
    entry = tk.Entry(win, textvariable=model_var, width=50)
    entry.pack(padx=10)

    def browse():
        path = ask_open_file()
        if path:
            model_var.set(path)

    tk.Button(win, text="Browse", command=browse).pack(pady=5)

    def save():
        settings["model_path"] = model_var.get()
        save_settings(settings)
        messagebox.showinfo("Saved", "Settings saved.")
        win.destroy()

    tk.Button(win, text="💾 Save", command=save).pack(pady=10)

def show_about():
    messagebox.showinfo(
        "ℹ️ About",
        f"🧬 HemoCnn\n"
        f"Version: {VERSION}\n"
        f"Author: Ganesh Kishore\n\n"
        "PyTorch-based CNN for blood cell classification\n"
        "using MobileNetV2."
    )

root = tk.Tk()
root.title("🧬 HemoCnn – Blood Cell Classifier")
root.geometry("360x520")
root.protocol("WM_DELETE_WINDOW", root.destroy)

tk.Label(root, text="🧬 HemoCnn", font=("Arial", 20, "bold")).pack(pady=(15, 5))
tk.Label(root, text="Blood Cell Classification Tool", fg="gray").pack(pady=(0, 5))

tk.Button(
    root,
    text="🧪 Test Cell",
    font=("Arial", 12),
    width=20,
    command=test_cell
).pack(pady=10)

try:
    logo_path = os.path.join(get_assets_dir(), "GK_productions_logo.png")
    img = Image.open(logo_path).resize((350, 350))
    logo = ImageTk.PhotoImage(img)
    lbl = tk.Label(root, image=logo, bg="white")
    lbl.image = logo
    lbl.pack(expand=True)
except Exception as e:
    print("Logo not loaded:", e)

bottom = tk.Frame(root)
bottom.pack(fill="x", padx=10, pady=(0, 10))

tk.Button(bottom, text="🛠 Settings", command=show_settings).pack(side="right", padx=5)
tk.Button(bottom, text="ℹ️ Info", command=show_about).pack(side="right", padx=5)

root.mainloop()
