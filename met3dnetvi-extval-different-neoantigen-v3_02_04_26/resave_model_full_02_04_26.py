"""
resave_model_full.py
────────────────────
Run this ONE script on your training machine (where best_model_v7.pt was created).
It converts the state_dict checkpoint to a full model object that can be loaded
on Kaggle with torch.load() without needing the model class definition.

Usage:
    python resave_model_full.py

Output:
    best_model_v7_full.pt   (upload this to your Kaggle dataset)
"""

import torch
import os

CHECKPOINT_IN  = "best_model_v7.pt"        # your existing file
CHECKPOINT_OUT = "best_model_v7_full.pt"   # what to upload to Kaggle

# ── Step 1: Load the existing state dict ─────────────────────────────────────
print(f"Loading: {CHECKPOINT_IN}")
ckpt = torch.load(CHECKPOINT_IN, map_location="cpu", weights_only=False)

if hasattr(ckpt, "state_dict"):
    print("Already a full model — just re-saving as best_model_v7_full.pt")
    torch.save(ckpt, CHECKPOINT_OUT)
    print(f"Done: {CHECKPOINT_OUT}")
    exit()

sd = ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt \
    else ckpt["model_state_dict"]

print(f"  State dict: {len(sd)} keys")
for k, v in list(sd.items())[:8]:
    shp = str(tuple(v.shape)) if hasattr(v, "shape") else str(v)
    print(f"  {k:<50} {shp}")

# ── Step 2: Reconstruct the model from your training code ─────────────────────
# PASTE YOUR MODEL CLASS IMPORT HERE:
# from model import Met3DNetVI          # <-- replace with your actual import
# model = Met3DNetVI(...)               # <-- replace with your constructor call

# If you have the training code available, use this pattern:
try:
    # Try importing from the most common locations
    import sys
    for path in [".", "..", "src", "models", "core"]:
        sys.path.insert(0, path)

    model = None
    for cls_name in ["Met3DNetVI", "MetNet", "ImmunogenicityGNN", "NeoAntigenGNN"]:
        for module_name in ["model", "models", "met3dnet", "gnn", "network"]:
            try:
                mod = __import__(module_name)
                cls = getattr(mod, cls_name)
                # Try common constructor signatures
                for args in [(), (10,), (11, 64, 4), (11, 64, 4, 10)]:
                    try:
                        model = cls(*args)
                        break
                    except: pass
                if model: break
            except: pass
        if model: break

    if model is None:
        raise ImportError("Could not find model class automatically")

    model.load_state_dict(sd, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model reconstructed: {type(model).__name__} | params={n_params:,}")
    torch.save(model, CHECKPOINT_OUT)
    print(f"✓ Saved full model: {CHECKPOINT_OUT}")
    print(f"  Upload this file to your Kaggle dataset.")

except Exception as e:
    print(f"\n⚠ Automatic reconstruction failed: {e}")
    print("""
MANUAL STEPS (2 minutes):
─────────────────────────
In your training script or notebook, find where you trained the model.
Add these two lines right after training completes:

    # Add after: model = Met3DNetVI(...)  /  after training loop
    torch.save(model, 'best_model_v7_full.pt')   # full model object
    print("Saved: best_model_v7_full.pt")

Then run training once more (or just load the state dict and resave):

    from model import Met3DNetVI   # your actual import
    model = Met3DNetVI(...)        # your actual constructor
    model.load_state_dict(torch.load('best_model_v7.pt', weights_only=False))
    model.eval()
    torch.save(model, 'best_model_v7_full.pt')

Upload best_model_v7_full.pt to Kaggle → your dataset → Add file.
Then update MODEL_PATH in Cell 0 of the notebook to point to the new file.
""")
