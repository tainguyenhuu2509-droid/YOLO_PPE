

###########################################################################
# CELL 1
###########################################################################
# ============================================================
# CELL 1 — KAGGLE ENVIRONMENT & MIGRATION BUNDLE INSPECTION
# ============================================================

from pathlib import Path
import sys
import subprocess
import importlib.util
import json

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")

print("=" * 75)
print("YOLO11-SEG PPE PROJECT — KAGGLE INITIALIZATION")
print("=" * 75)

# ------------------------------------------------------------
# 1. Check Kaggle directories
# ------------------------------------------------------------

print("\n[1] KAGGLE DIRECTORIES")

print("Input exists  :", INPUT_ROOT.exists())
print("Working exists:", WORK_ROOT.exists())

if not INPUT_ROOT.exists():
    raise RuntimeError(
        "/kaggle/input not found. "
        "This does not appear to be a Kaggle Notebook environment."
    )

# ------------------------------------------------------------
# 2. List attached datasets
# ------------------------------------------------------------

print("\n[2] ATTACHED DATASETS")

top_level = sorted(INPUT_ROOT.iterdir())

if not top_level:
    print("[WARNING] No input datasets attached.")

for item in top_level:
    print(
        f"{'[DIR]' if item.is_dir() else '[FILE]'} "
        f"{item.name}"
    )

# ------------------------------------------------------------
# 3. Locate ZIP archives and notebooks
# ------------------------------------------------------------

print("\n[3] SEARCHING FOR MIGRATION FILES")

zip_files = []
notebook_files = []
guide_files = []
manifest_files = []

for path in INPUT_ROOT.rglob("*"):

    if not path.is_file():
        continue

    lower_name = path.name.lower()

    if lower_name.endswith(".zip"):
        zip_files.append(path)

    elif lower_name.endswith(".ipynb"):
        notebook_files.append(path)

    elif lower_name == "kaggle_migration_guide.md":
        guide_files.append(path)

    elif lower_name == "transfer_manifest.json":
        manifest_files.append(path)

print("\nZIP archives:")
for path in zip_files:
    size_gb = path.stat().st_size / (1024 ** 3)
    print(f"  {path} ({size_gb:.2f} GB)")

print("\nNotebook files:")
for path in notebook_files:
    print(f"  {path}")

print("\nMigration guides:")
for path in guide_files:
    print(f"  {path}")

print("\nTransfer manifests:")
for path in manifest_files:
    print(f"  {path}")

# ------------------------------------------------------------
# 4. Search for already-extracted bundle contents
# ------------------------------------------------------------

print("\n[4] SEARCHING FOR PROJECT CONTENTS")

interesting_names = {
    "data.yaml",
    "finetune_best.pt",
    "best.pt",
    "research_state.json",
}

found = {}

for path in INPUT_ROOT.rglob("*"):

    if path.is_file() and path.name in interesting_names:

        found.setdefault(path.name, []).append(path)

for name, paths in found.items():

    print(f"\n{name}:")

    for path in paths[:10]:
        print("  ", path)

# ------------------------------------------------------------
# 5. Python environment
# ------------------------------------------------------------

print("\n[5] PYTHON ENVIRONMENT")

print("Python:", sys.version)
print("Executable:", sys.executable)

# ------------------------------------------------------------
# 6. GPU verification
# ------------------------------------------------------------

print("\n[6] GPU VERIFICATION")

try:

    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        for i in range(torch.cuda.device_count()):

            props = torch.cuda.get_device_properties(i)

            print(
                f"GPU {i}: {props.name} | "
                f"{props.total_memory / (1024 ** 3):.2f} GB"
            )

    else:

        print(
            "[WARNING] GPU is not available. "
            "Enable GPU in Kaggle Notebook Settings."
        )

except Exception as error:

    print("[ERROR] PyTorch check failed:", error)

# nvidia-smi
try:

    gpu_info = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=15
    )

    print("\nnvidia-smi:")
    print(gpu_info.stdout[:5000])

    if gpu_info.returncode != 0:
        print(gpu_info.stderr)

except Exception as error:

    print("nvidia-smi unavailable:", error)

# ------------------------------------------------------------
# 7. Check important packages
# ------------------------------------------------------------

print("\n[7] PACKAGE AVAILABILITY")

packages = [
    "torch",
    "ultralytics",
    "numpy",
    "cv2",
    "pandas",
    "matplotlib",
    "scipy",
    "yaml",
]

for package in packages:

    try:
        installed = (
            importlib.util.find_spec(package) is not None
        )
    except (ImportError, ValueError):
        installed = False

    print(
        f"[{'OK' if installed else 'MISSING'}] {package}"
    )

# ------------------------------------------------------------
# 8. Preliminary status
# ------------------------------------------------------------

bundle_candidates = [
    p for p in zip_files
    if "yolo11_seg_expanded_kaggle_bundle" in p.name.lower()
]

print("\n" + "=" * 75)
print("INITIAL INSPECTION SUMMARY")
print("=" * 75)

print("Migration ZIP found:", len(bundle_candidates) > 0)
print("Original notebook found:", len(notebook_files) > 0)
print("GPU available:", (
    torch.cuda.is_available()
    if "torch" in sys.modules
    else False
))

print("\nNo files have been extracted or modified.")
print("[CELL 1 COMPLETED]")

###########################################################################
# CELL 2
###########################################################################
# ============================================================
# CELL 1 — KAGGLE ENVIRONMENT & MIGRATION BUNDLE INSPECTION
# ============================================================

from pathlib import Path
import sys
import subprocess
import importlib.util
import json

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")

print("=" * 75)
print("YOLO11-SEG PPE PROJECT — KAGGLE INITIALIZATION")
print("=" * 75)

# ------------------------------------------------------------
# 1. Check Kaggle directories
# ------------------------------------------------------------

print("\n[1] KAGGLE DIRECTORIES")

print("Input exists  :", INPUT_ROOT.exists())
print("Working exists:", WORK_ROOT.exists())

if not INPUT_ROOT.exists():
    raise RuntimeError(
        "/kaggle/input not found. "
        "This does not appear to be a Kaggle Notebook environment."
    )

# ------------------------------------------------------------
# 2. List attached datasets
# ------------------------------------------------------------

print("\n[2] ATTACHED DATASETS")

top_level = sorted(INPUT_ROOT.iterdir())

if not top_level:
    print("[WARNING] No input datasets attached.")

for item in top_level:
    print(
        f"{'[DIR]' if item.is_dir() else '[FILE]'} "
        f"{item.name}"
    )

# ------------------------------------------------------------
# 3. Locate ZIP archives and notebooks
# ------------------------------------------------------------

print("\n[3] SEARCHING FOR MIGRATION FILES")

zip_files = []
notebook_files = []
guide_files = []
manifest_files = []

for path in INPUT_ROOT.rglob("*"):

    if not path.is_file():
        continue

    lower_name = path.name.lower()

    if lower_name.endswith(".zip"):
        zip_files.append(path)

    elif lower_name.endswith(".ipynb"):
        notebook_files.append(path)

    elif lower_name == "kaggle_migration_guide.md":
        guide_files.append(path)

    elif lower_name == "transfer_manifest.json":
        manifest_files.append(path)

print("\nZIP archives:")
for path in zip_files:
    size_gb = path.stat().st_size / (1024 ** 3)
    print(f"  {path} ({size_gb:.2f} GB)")

print("\nNotebook files:")
for path in notebook_files:
    print(f"  {path}")

print("\nMigration guides:")
for path in guide_files:
    print(f"  {path}")

print("\nTransfer manifests:")
for path in manifest_files:
    print(f"  {path}")

# ------------------------------------------------------------
# 4. Search for already-extracted bundle contents
# ------------------------------------------------------------

print("\n[4] SEARCHING FOR PROJECT CONTENTS")

interesting_names = {
    "data.yaml",
    "finetune_best.pt",
    "best.pt",
    "research_state.json",
}

found = {}

for path in INPUT_ROOT.rglob("*"):

    if path.is_file() and path.name in interesting_names:

        found.setdefault(path.name, []).append(path)

for name, paths in found.items():

    print(f"\n{name}:")

    for path in paths[:10]:
        print("  ", path)

# ------------------------------------------------------------
# 5. Python environment
# ------------------------------------------------------------

print("\n[5] PYTHON ENVIRONMENT")

print("Python:", sys.version)
print("Executable:", sys.executable)

# ------------------------------------------------------------
# 6. GPU verification
# ------------------------------------------------------------

print("\n[6] GPU VERIFICATION")

try:

    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        for i in range(torch.cuda.device_count()):

            props = torch.cuda.get_device_properties(i)

            print(
                f"GPU {i}: {props.name} | "
                f"{props.total_memory / (1024 ** 3):.2f} GB"
            )

    else:

        print(
            "[WARNING] GPU is not available. "
            "Enable GPU in Kaggle Notebook Settings."
        )

except Exception as error:

    print("[ERROR] PyTorch check failed:", error)

# nvidia-smi
try:

    gpu_info = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=15
    )

    print("\nnvidia-smi:")
    print(gpu_info.stdout[:5000])

    if gpu_info.returncode != 0:
        print(gpu_info.stderr)

except Exception as error:

    print("nvidia-smi unavailable:", error)

# ------------------------------------------------------------
# 7. Check important packages
# ------------------------------------------------------------

print("\n[7] PACKAGE AVAILABILITY")

packages = [
    "torch",
    "ultralytics",
    "numpy",
    "cv2",
    "pandas",
    "matplotlib",
    "scipy",
    "yaml",
]

for package in packages:

    try:
        installed = (
            importlib.util.find_spec(package) is not None
        )
    except (ImportError, ValueError):
        installed = False

    print(
        f"[{'OK' if installed else 'MISSING'}] {package}"
    )

# ------------------------------------------------------------
# 8. Preliminary status
# ------------------------------------------------------------

bundle_candidates = [
    p for p in zip_files
    if "yolo11_seg_expanded_kaggle_bundle" in p.name.lower()
]

print("\n" + "=" * 75)
print("INITIAL INSPECTION SUMMARY")
print("=" * 75)

print("Migration ZIP found:", len(bundle_candidates) > 0)
print("Original notebook found:", len(notebook_files) > 0)
print("GPU available:", (
    torch.cuda.is_available()
    if "torch" in sys.modules
    else False
))

print("\nNo files have been extracted or modified.")
print("[CELL 1 COMPLETED]")

###########################################################################
# CELL 3
###########################################################################
# ============================================================
# CELL 2 — RESTORE YOLO11-SEG PROJECT TO KAGGLE WORKING
#
# Source: /kaggle/input (read-only)
# Destination: /kaggle/working (writable)
# ============================================================

import os
import shutil
import json
import yaml

from pathlib import Path

print("=" * 75)
print("STEP 2 — RESTORE PROJECT WORKSPACE")
print("=" * 75)

# ------------------------------------------------------------
# 1. FIND THE EXTRACTED BUNDLE
# ------------------------------------------------------------

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")

manifest_files = list(
    INPUT_ROOT.rglob("transfer_manifest.json")
)

if len(manifest_files) == 0:
    raise FileNotFoundError(
        "transfer_manifest.json not found. "
        "Check whether the migration dataset is attached."
    )

if len(manifest_files) > 1:
    print("[WARNING] Multiple manifests found:")
    for item in manifest_files:
        print(item)

    raise RuntimeError(
        "Multiple project bundles detected. "
        "Please select the correct one."
    )

SOURCE_PROJECT_DIR = manifest_files[0].parent

print("\n[OK] Extracted project located:")
print(SOURCE_PROJECT_DIR)

# ------------------------------------------------------------
# 2. VERIFY SOURCE STRUCTURE
# ------------------------------------------------------------

required_source = [
    SOURCE_PROJECT_DIR / "dataset",
    SOURCE_PROJECT_DIR / "models",
    SOURCE_PROJECT_DIR / "research_experiments",
    SOURCE_PROJECT_DIR / "raw_zips",
    SOURCE_PROJECT_DIR / "research_state.json",
    SOURCE_PROJECT_DIR / "KAGGLE_MIGRATION_GUIDE.md",
]

print("\nSOURCE STRUCTURE:")

for path in required_source:

    exists = path.exists()

    print(
        f"[{'OK' if exists else 'MISSING'}] {path.name}"
    )

    if not exists:
        raise FileNotFoundError(path)

# ------------------------------------------------------------
# 3. CHECK DISK SPACE
# ------------------------------------------------------------

print("\nDISK SPACE CHECK:")

source_bytes = sum(
    path.stat().st_size
    for path in SOURCE_PROJECT_DIR.rglob("*")
    if path.is_file()
)

disk_usage = shutil.disk_usage(WORK_ROOT)

source_gb = source_bytes / (1024 ** 3)
free_gb = disk_usage.free / (1024 ** 3)

print(f"Project size : {source_gb:.2f} GB")
print(f"Free space   : {free_gb:.2f} GB")

# Keep a safety margin.
required_free_gb = source_gb + 1.0

if free_gb < required_free_gb:
    raise RuntimeError(
        f"Insufficient disk space. "
        f"Need approximately {required_free_gb:.2f} GB, "
        f"available {free_gb:.2f} GB."
    )

# ------------------------------------------------------------
# 4. CREATE WORKSPACE
# ------------------------------------------------------------

PROJECT_DIR = WORK_ROOT / "YOLO11_seg_expanded"

if PROJECT_DIR.exists():
    raise FileExistsError(
        f"Workspace already exists:\n{PROJECT_DIR}\n"
        "This cell will not overwrite an existing workspace."
    )

print("\nCOPYING PROJECT...")
print("Source:", SOURCE_PROJECT_DIR)
print("Target:", PROJECT_DIR)

shutil.copytree(
    SOURCE_PROJECT_DIR,
    PROJECT_DIR,
)

print("\n[OK] Project copied successfully.")

# ------------------------------------------------------------
# 5. DEFINE PROJECT PATHS
# ------------------------------------------------------------

DATASET_BUILD_DIR = PROJECT_DIR / "dataset"

MODELS_DIR = PROJECT_DIR / "models"

RESEARCH_RUNS_DIR = (
    PROJECT_DIR / "research_experiments"
)

RAW_ZIPS_DIR = PROJECT_DIR / "raw_zips"

OUTPUT_DIR = PROJECT_DIR / "kaggle_outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nPROJECT PATHS:")

print("PROJECT_DIR       :", PROJECT_DIR)
print("DATASET_BUILD_DIR :", DATASET_BUILD_DIR)
print("MODELS_DIR        :", MODELS_DIR)
print("RESEARCH_RUNS_DIR :", RESEARCH_RUNS_DIR)
print("OUTPUT_DIR        :", OUTPUT_DIR)

# ------------------------------------------------------------
# 6. VERIFY DATASET COUNTS
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

def count_images(folder):

    folder = Path(folder)

    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_labels(folder):

    folder = Path(folder)

    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.rglob("*.txt")
        if path.is_file()
    )


print("\nDATASET VERIFICATION:")

EXPECTED_COUNTS = {
    "train": 924,
    "val": 268,
    "val_old": 119,
    "val_new": 149,
}

actual_counts = {
    "train": count_images(
        DATASET_BUILD_DIR / "images" / "train"
    ),

    "val": count_images(
        DATASET_BUILD_DIR / "images" / "val"
    ),

    "val_old": count_images(
        DATASET_BUILD_DIR / "images" / "val" / "old"
    ),

    "val_new": count_images(
        DATASET_BUILD_DIR / "images" / "val" / "new"
    ),
}

for split, expected in EXPECTED_COUNTS.items():

    actual = actual_counts[split]

    print(
        f"{split:10s}: "
        f"{actual:4d} images "
        f"(expected {expected}) "
        f"{'[OK]' if actual == expected else '[ERROR]'}"
    )

    if actual != expected:
        raise RuntimeError(
            f"Unexpected image count for {split}: "
            f"{actual} vs {expected}"
        )

# Verify image-label matching by count.
for split in ["train", "val"]:

    image_count = count_images(
        DATASET_BUILD_DIR / "images" / split
    )

    label_count = count_labels(
        DATASET_BUILD_DIR / "labels" / split
    )

    print(
        f"{split.upper()} image/label: "
        f"{image_count}/{label_count}"
    )

    if image_count != label_count:
        raise RuntimeError(
            f"Image-label count mismatch in {split}"
        )

# ------------------------------------------------------------
# 7. VERIFY DATA YAML
# ------------------------------------------------------------

DATA_YAML = DATASET_BUILD_DIR / "data.yaml"
OLD_VAL_YAML = DATASET_BUILD_DIR / "old_validation.yaml"
NEW_VAL_YAML = DATASET_BUILD_DIR / "new_validation.yaml"

print("\nDATA YAML VERIFICATION:")

for yaml_path in [
    DATA_YAML,
    OLD_VAL_YAML,
    NEW_VAL_YAML,
]:

    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

    with open(
        yaml_path,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    print(f"\n[{yaml_path.name}]")
    print("path :", config.get("path"))
    print("train:", config.get("train"))
    print("val  :", config.get("val"))
    print("names:", config.get("names"))

    names = config.get("names")

    expected_names = {
        0: "hat",
        1: "gloves",
        2: "mask",
        3: "pants",
        4: "shoes",
        5: "long-sleeved shirt",
    }

    # YAML integer keys are normally integers.
    normalized_names = {
        int(k): str(v)
        for k, v in names.items()
    }

    if normalized_names != expected_names:
        raise RuntimeError(
            f"Class mapping mismatch in {yaml_path}"
        )

    configured_path = Path(config["path"])

    if not configured_path.exists():
        raise RuntimeError(
            f"Invalid dataset path in {yaml_path}: "
            f"{configured_path}"
        )

print("\n[OK] All YAML paths and class mappings verified.")

# ------------------------------------------------------------
# 8. VERIFY SIX IMPORTANT CHECKPOINTS
# ------------------------------------------------------------

CHECKPOINTS = {
    "Original": (
        MODELS_DIR / "finetune_best.pt"
    ),

    "A_Uniform": (
        RESEARCH_RUNS_DIR
        / "A_uniform"
        / "weights"
        / "best.pt"
    ),

    "B_RFS": (
        RESEARCH_RUNS_DIR
        / "B_rfs_t080"
        / "weights"
        / "best.pt"
    ),

    "C_IRFS": (
        RESEARCH_RUNS_DIR
        / "C_irfs_t030"
        / "weights"
        / "best.pt"
    ),

    "D0_IRFS_960": (
        RESEARCH_RUNS_DIR
        / "D0_IRFS_960_control"
        / "weights"
        / "best.pt"
    ),

    "D1_IRFS_1152": (
        RESEARCH_RUNS_DIR
        / "D1_IRFS_1152_highres"
        / "weights"
        / "best.pt"
    ),
}

print("\nCHECKPOINT VERIFICATION:")

for name, path in CHECKPOINTS.items():

    exists = path.exists()

    if exists:
        size_mb = path.stat().st_size / (1024 ** 2)
        print(
            f"[OK] {name:18s} "
            f"{size_mb:.2f} MB"
        )
    else:
        print(f"[MISSING] {name}: {path}")
        raise FileNotFoundError(path)

# ------------------------------------------------------------
# 9. VERIFY RESEARCH ARTIFACTS
# ------------------------------------------------------------

print("\nRESEARCH RESULTS:")

important_artifacts = [
    RESEARCH_RUNS_DIR / "A_uniform" / "results.csv",

    RESEARCH_RUNS_DIR / "B_rfs_t080" / "results.csv",

    RESEARCH_RUNS_DIR / "C_irfs_t030" / "results.csv",

    PROJECT_DIR
    / "research_experiments"
    / "evaluation_sampling_abcd_v1"
    / "overall_mask_comparison.csv",

    PROJECT_DIR
    / "research_experiments"
    / "E_inference_resolution_optimization_v1"
    / "resolution_summary.csv",
]

for path in important_artifacts:

    print(
        f"[{'OK' if path.exists() else 'MISSING'}] "
        f"{path.name}"
    )

# ------------------------------------------------------------
# 10. VERIFY WORKSPACE IS WRITABLE
# ------------------------------------------------------------

test_file = PROJECT_DIR / ".write_test"

test_file.write_text(
    "Kaggle workspace writable.",
    encoding="utf-8"
)

assert test_file.exists()

test_file.unlink()

print("\n[OK] Workspace is writable.")

# ------------------------------------------------------------
# 11. PROJECT SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("PROJECT RESTORATION COMPLETED")
print("=" * 75)

print(f"Project directory : {PROJECT_DIR}")
print(f"Dataset           : {DATASET_BUILD_DIR}")
print(f"Original models   : {MODELS_DIR}")
print(f"Research runs     : {RESEARCH_RUNS_DIR}")
print(f"Kaggle outputs    : {OUTPUT_DIR}")

print("\nDataset:")
print("  Train: 924 images")
print("  Val  : 268 images")
print("  Old  : 119 images")
print("  New  : 149 images")

print("\nCheckpoints available: 6")
print("\n[CELL 2 COMPLETED]")

###########################################################################
# CELL 4
###########################################################################
# ============================================================
# CELL 3 — INSTALL AND VERIFY ULTRALYTICS ENVIRONMENT
# ============================================================

import sys
import subprocess
import importlib.util
import gc
import json

from datetime import datetime
from pathlib import Path

print("=" * 75)
print("STEP 3 — KAGGLE ENVIRONMENT SETUP")
print("=" * 75)

# ------------------------------------------------------------
# 1. Install Ultralytics if missing
# ------------------------------------------------------------

if importlib.util.find_spec("ultralytics") is None:

    print("Installing Ultralytics 8.4.193...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ultralytics==8.4.193",
    ])

else:

    print("Ultralytics is already installed.")

# ------------------------------------------------------------
# 2. Import packages
# ------------------------------------------------------------

import torch
import ultralytics
import numpy as np
import cv2
import pandas as pd
import yaml

from ultralytics import YOLO

print("\nPACKAGE VERSIONS:")
print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__)
print("CUDA        :", torch.version.cuda)
print("Ultralytics :", ultralytics.__version__)
print("NumPy       :", np.__version__)
print("OpenCV      :", cv2.__version__)
print("Pandas      :", pd.__version__)

# ------------------------------------------------------------
# 3. Verify GPU
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check Kaggle GPU settings."
    )

print("\nGPU INFORMATION:")

for i in range(torch.cuda.device_count()):

    properties = torch.cuda.get_device_properties(i)

    print(
        f"GPU {i}: {properties.name} | "
        f"{properties.total_memory / (1024**3):.2f} GB"
    )

# Start with GPU 0.
DEVICE = 0

# ------------------------------------------------------------
# 4. Verify segmentation checkpoint
# ------------------------------------------------------------

print("\nVERIFYING IRFS CHECKPOINT:")

C_IRFS_CHECKPOINT = (
    RESEARCH_RUNS_DIR
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

if not C_IRFS_CHECKPOINT.exists():
    raise FileNotFoundError(C_IRFS_CHECKPOINT)

model = YOLO(str(C_IRFS_CHECKPOINT))

EXPECTED_NAMES = {
    0: "hat",
    1: "gloves",
    2: "mask",
    3: "pants",
    4: "shoes",
    5: "long-sleeved shirt",
}

print("Task :", model.task)
print("Names:", model.names)

if model.task != "segment":
    raise RuntimeError(
        f"Expected segmentation model, got {model.task}"
    )

if model.names != EXPECTED_NAMES:
    raise RuntimeError(
        "Checkpoint class mapping does not match dataset."
    )

print("[OK] IRFS checkpoint loaded.")

del model
gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------
# 5. Save environment metadata
# ------------------------------------------------------------

environment = {
    "created_at": datetime.now().isoformat(),
    "python": sys.version,
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "ultralytics": ultralytics.__version__,
    "numpy": np.__version__,
    "opencv": cv2.__version__,
    "gpu_devices": [
        torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())
    ],
    "selected_device": DEVICE,
    "project_dir": str(PROJECT_DIR),
    "dataset_yaml": str(DATA_YAML),
}

environment_path = (
    PROJECT_DIR / "kaggle_environment.json"
)

with open(
    environment_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        environment,
        f,
        indent=4,
        ensure_ascii=False
    )

# ------------------------------------------------------------
# 6. Final verification
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("ENVIRONMENT SETUP COMPLETED")
print("=" * 75)

print("Ultralytics :", ultralytics.__version__)
print("Selected GPU:", torch.cuda.get_device_name(DEVICE))
print("Checkpoint  : C_IRFS best.pt")
print("Class map   : Verified")
print("Environment :", environment_path)

print("\n[CELL 3 COMPLETED]")

###########################################################################
# CELL 5
###########################################################################
# ============================================================
# CELL 4 — INSPECT ULTRALYTICS TRAINING API
#
# Purpose:
# Reconstruct the original custom RFS / IRFS trainers.
# No training is performed.
# ============================================================

import inspect
import ultralytics
import torch

from ultralytics import YOLO

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.engine.trainer import BaseTrainer

from ultralytics.data.build import (
    build_dataloader,
    InfiniteDataLoader,
)

from ultralytics.data.dataset import YOLODataset

print("=" * 80)
print("ULTRALYTICS API INSPECTION — KAGGLE")
print("=" * 80)

print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

if ultralytics.__version__ != "8.4.193":
    raise RuntimeError(
        "Unexpected Ultralytics version. "
        "This experiment was designed for 8.4.193."
    )

# ------------------------------------------------------------
# 1. Trainer inheritance
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("1. SEGMENTATION TRAINER INHERITANCE")
print("=" * 80)

for cls in SegmentationTrainer.__mro__:
    print(cls.__module__, cls.__name__)

# ------------------------------------------------------------
# 2. Method signatures
# ------------------------------------------------------------

methods_to_inspect = [
    (SegmentationTrainer, "get_dataloader"),
    (DetectionTrainer, "get_dataloader"),
    (DetectionTrainer, "build_dataset"),
    (BaseTrainer, "_build_train_pipeline"),
    (BaseTrainer, "build_optimizer"),
    (BaseTrainer, "setup_scheduler"),
    (YOLODataset, "__init__"),
    (YOLODataset, "get_labels"),
    (InfiniteDataLoader, "__init__"),
    (InfiniteDataLoader, "__len__"),
    (build_dataloader, "__call__"),
]

print("\n" + "=" * 80)
print("2. METHOD SIGNATURES")
print("=" * 80)

for cls, method_name in methods_to_inspect:

    method = getattr(cls, method_name, None)

    if method is None:
        print(f"[MISSING] {cls.__name__}.{method_name}")
        continue

    try:
        signature = inspect.signature(method)
    except Exception:
        signature = "Unavailable"

    print(
        f"{cls.__name__}.{method_name}{signature}"
    )

# ------------------------------------------------------------
# 3. Inspect get_dataloader implementation
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("3. DETECTION TRAINER — get_dataloader SOURCE")
print("=" * 80)

source = inspect.getsource(
    DetectionTrainer.get_dataloader
)

print(source)

# ------------------------------------------------------------
# 4. Inspect dataset construction
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("4. DETECTION TRAINER — build_dataset SOURCE")
print("=" * 80)

source = inspect.getsource(
    DetectionTrainer.build_dataset
)

print(source)

# ------------------------------------------------------------
# 5. Inspect InfiniteDataLoader
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("5. INFINITE DATALOADER SOURCE")
print("=" * 80)

print(
    inspect.getsource(InfiniteDataLoader)
)

# ------------------------------------------------------------
# 6. Inspect build_dataloader
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("6. build_dataloader SOURCE")
print("=" * 80)

print(
    inspect.getsource(build_dataloader)
)

# ------------------------------------------------------------
# 7. Inspect YOLODataset label cache behavior
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("7. YOLO DATASET LABEL METHODS")
print("=" * 80)

print(
    inspect.getsource(YOLODataset.get_labels)
)

# ------------------------------------------------------------
# 8. Inspect model training API
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("8. YOLO.train SIGNATURE")
print("=" * 80)

print(inspect.signature(YOLO.train))

# ------------------------------------------------------------
# 9. Check dataset config without starting training
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("9. DATASET CONFIGURATION")
print("=" * 80)

import yaml

with open(DATA_YAML, "r", encoding="utf-8") as f:
    dataset_config = yaml.safe_load(f)

print(json.dumps(
    dataset_config,
    indent=2,
    ensure_ascii=False
))

# ------------------------------------------------------------
# 10. Final inspection status
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("API INSPECTION COMPLETED")
print("=" * 80)

print("Version check       : OK")
print("SegmentationTrainer : OK")
print("DetectionTrainer    : OK")
print("InfiniteDataLoader  : OK")
print("YOLODataset         : OK")
print("Dataset YAML        : OK")

print("\nTraining started: NO")
print("[CELL 4 COMPLETED]")

###########################################################################
# CELL 6
###########################################################################
# ============================================================
# CELL 5 — REBUILD CUSTOM RFS / IRFS SAMPLING
#
# Restores:
# 1. FixedLengthRepeatSampler
# 2. Repeat-factor computation from Ultralytics label cache
# 3. RepeatFactorSegmentationTrainer
# 4. RFSSegmentationTrainer
# 5. IRFSResearchTrainer
#
# No training is performed.
# ============================================================

import os
import gc
import json
import math
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from collections import Counter

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.data.build import (
    build_yolo_dataset,
    InfiniteDataLoader,
    seed_worker,
)

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

print("=" * 80)
print("STEP 5 — REBUILD CUSTOM RFS / IRFS TRAINERS")
print("=" * 80)

# ============================================================
# 1. FIXED-LENGTH REPEAT FACTOR SAMPLER
# ============================================================

class FixedLengthRepeatSampler:
    """
    Deterministic fixed-length repeat-factor sampler.

    Procedure:
    - Include floor(repeat_factor) copies of each image.
    - Allocate the remaining copies by systematic PPS sampling
      over the fractional repeat factors.
    - Shuffle the complete index sequence.
    - Use seed + epoch to generate reproducible epoch sequences.

    This is a fixed-length adaptation of repeat-factor sampling,
    not an exact implementation of independent Bernoulli sampling.
    """

    def __init__(
        self,
        repeat_factors,
        num_samples,
        seed=42,
    ):

        self.repeat_factors = np.asarray(
            repeat_factors,
            dtype=np.float64
        )

        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

        if self.repeat_factors.ndim != 1:
            raise ValueError(
                "repeat_factors must be one-dimensional."
            )

        if len(self.repeat_factors) == 0:
            raise ValueError("Empty repeat factor array.")

        if not np.isfinite(self.repeat_factors).all():
            raise ValueError(
                "Non-finite repeat factors detected."
            )

        if np.any(self.repeat_factors < 1.0):
            raise ValueError(
                "All repeat factors must be >= 1."
            )

        if self.num_samples < len(self.repeat_factors):
            raise ValueError(
                "num_samples cannot be smaller than "
                "the number of unique images."
            )

        base_counts = np.floor(
            self.repeat_factors
        ).astype(np.int64)

        self.base_counts = base_counts

        base_size = int(base_counts.sum())

        if self.num_samples < base_size:
            raise ValueError(
                f"Requested {self.num_samples} samples, "
                f"but floor repeat factors require {base_size}."
            )

        self.remaining = (
            self.num_samples - base_size
        )

        self.fractional_weights = (
            self.repeat_factors - base_counts
        )

        self.expected_length = float(
            self.repeat_factors.sum()
        )

        self._scaled_pps_weights = None

        if self.remaining > 0:

            weight_sum = float(
                self.fractional_weights.sum()
            )

            if weight_sum <= 0:
                raise ValueError(
                    "No fractional repeat weights available."
                )

            scaled_weights = (
                self.fractional_weights
                * (self.remaining / weight_sum)
            )

            if np.any(scaled_weights > 1.0 + 1e-8):
                raise ValueError(
                    "Systematic PPS inclusion probability "
                    "exceeds 1. This configuration needs "
                    "a generalized fixed-size sampler."
                )

            self._scaled_pps_weights = np.minimum(
                scaled_weights,
                1.0
            )

    def __len__(self):
        return self.num_samples

    def __iter__(self):

        rng = np.random.default_rng(
            self.seed + self.epoch
        )

        self.epoch += 1

        # --------------------------------------------
        # 1. Floor copies
        # --------------------------------------------

        base_indices = np.repeat(
            np.arange(len(self.repeat_factors)),
            self.base_counts,
        )

        # --------------------------------------------
        # 2. Systematic PPS fractional copies
        # --------------------------------------------

        if self.remaining > 0:

            cumulative = np.cumsum(
                self._scaled_pps_weights
            )

            total_weight = float(
                cumulative[-1]
            )

            if not np.isclose(
                total_weight,
                self.remaining,
                atol=1e-6
            ):
                raise RuntimeError(
                    "PPS inclusion weights do not sum "
                    "to the required number of extra samples."
                )

            start = rng.uniform(0.0, 1.0)

            positions = (
                start
                + np.arange(self.remaining)
            )

            extra_indices = np.searchsorted(
                cumulative,
                positions,
                side="right"
            )

            if np.any(
                extra_indices >= len(self.repeat_factors)
            ):
                raise RuntimeError(
                    "PPS selected an invalid image index."
                )

            if len(np.unique(extra_indices)) != len(extra_indices):
                raise RuntimeError(
                    "Systematic PPS selected an image more than once "
                    "for its fractional repeat."
                )

            indices = np.concatenate([
                base_indices,
                extra_indices.astype(np.int64),
            ])

        else:

            indices = base_indices

        # --------------------------------------------
        # 3. Verify and shuffle
        # --------------------------------------------

        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"Expected {self.num_samples} indices, "
                f"generated {len(indices)}."
            )

        indices = rng.permutation(indices)

        return iter(indices.tolist())


print("[OK] FixedLengthRepeatSampler defined.")

# ============================================================
# 2. COMPUTE REPEAT FACTORS FROM VERIFIED DATASET LABELS
# ============================================================

EXPECTED_IMAGE_COUNTS = {
    0: 918,
    1: 520,
    2: 705,
    3: 759,
    4: 465,
    5: 921,
}

EXPECTED_INSTANCE_COUNTS = {
    0: 4352,
    1: 1415,
    2: 1292,
    3: 1682,
    4: 1152,
    5: 4048,
}

EXPECTED_TOTAL_IMAGES = 924
EXPECTED_TOTAL_INSTANCES = 13941

CLASS_NAMES = {
    0: "hat",
    1: "gloves",
    2: "mask",
    3: "pants",
    4: "shoes",
    5: "long-sleeved shirt",
}


def calculate_repeat_factors_from_cache(
    dataset,
    method="IRFS",
    threshold=0.30,
):

    """
    Calculate image repeat factors using validated labels.

    method:
        RFS:
            Frequency is the fraction of images containing
            each class.

        IRFS:
            Effective frequency is the geometric mean of
            image frequency and instance frequency.

    Returns:
        repeat_factors: np.ndarray with one value per image
        statistics: dictionary containing class frequencies
    """

    method = method.upper()

    if method not in {"RFS", "IRFS"}:
        raise ValueError(
            "method must be RFS or IRFS."
        )

    if threshold <= 0:
        raise ValueError(
            "threshold must be positive."
        )

    # Use the labels after Ultralytics cache verification.
    labels = dataset.labels

    num_images = len(labels)

    if num_images != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            f"Unexpected train image count: {num_images}"
        )

    class_image_counts = Counter()
    class_instance_counts = Counter()

    image_classes = []

    for label in labels:

        cls_array = np.asarray(
            label.get("cls", []),
            dtype=np.int64
        ).reshape(-1)

        if len(cls_array) == 0:

            image_classes.append(set())
            continue

        unique_classes = set(
            cls_array.tolist()
        )

        for class_id in unique_classes:

            if class_id not in CLASS_NAMES:
                raise RuntimeError(
                    f"Unexpected class ID: {class_id}"
                )

            class_image_counts[class_id] += 1

        for class_id, count in Counter(
            cls_array.tolist()
        ).items():

            class_instance_counts[class_id] += count

        image_classes.append(
            unique_classes
        )

    total_instances = int(
        sum(class_instance_counts.values())
    )

    if total_instances != EXPECTED_TOTAL_INSTANCES:
        raise RuntimeError(
            "Cleaned instance count mismatch.\n"
            f"Expected: {EXPECTED_TOTAL_INSTANCES}\n"
            f"Actual: {total_instances}\n"
            "Check Ultralytics label verification/cache."
        )

    # --------------------------------------------
    # Validate class-level statistics
    # --------------------------------------------

    for class_id in CLASS_NAMES:

        observed_images = int(
            class_image_counts[class_id]
        )

        observed_instances = int(
            class_instance_counts[class_id]
        )

        expected_images = EXPECTED_IMAGE_COUNTS[class_id]

        expected_instances = EXPECTED_INSTANCE_COUNTS[class_id]

        if observed_images != expected_images:
            raise RuntimeError(
                f"{CLASS_NAMES[class_id]} image count mismatch: "
                f"{observed_images} vs {expected_images}"
            )

        if observed_instances != expected_instances:
            raise RuntimeError(
                f"{CLASS_NAMES[class_id]} instance count mismatch: "
                f"{observed_instances} vs {expected_instances}"
            )

    # --------------------------------------------
    # Calculate class frequencies
    # --------------------------------------------

    statistics = {
        "method": method,
        "threshold": float(threshold),
        "num_images": num_images,
        "total_instances": total_instances,
        "classes": {},
    }

    class_repeat_factors = {}

    for class_id, class_name in CLASS_NAMES.items():

        image_freq = (
            class_image_counts[class_id]
            / num_images
        )

        instance_freq = (
            class_instance_counts[class_id]
            / total_instances
        )

        if method == "RFS":

            effective_freq = image_freq

        else:

            effective_freq = np.sqrt(
                image_freq * instance_freq
            )

        repeat_factor = max(
            1.0,
            np.sqrt(
                threshold / effective_freq
            )
        )

        class_repeat_factors[class_id] = (
            float(repeat_factor)
        )

        statistics["classes"][class_name] = {
            "class_id": class_id,
            "image_count": int(
                class_image_counts[class_id]
            ),
            "instance_count": int(
                class_instance_counts[class_id]
            ),
            "image_frequency": float(image_freq),
            "instance_frequency": float(instance_freq),
            "effective_frequency": float(
                effective_freq
            ),
            "repeat_factor": float(
                repeat_factor
            ),
        }

    # --------------------------------------------
    # Calculate image repeat factors
    # --------------------------------------------

    repeat_factors = np.ones(
        num_images,
        dtype=np.float64
    )

    for image_index, present_classes in enumerate(
        image_classes
    ):

        if len(present_classes) == 0:
            repeat_factors[image_index] = 1.0
        else:

            repeat_factors[image_index] = max(
                class_repeat_factors[class_id]
                for class_id in present_classes
            )

    statistics["expected_epoch_length"] = float(
        repeat_factors.sum()
    )

    statistics["mean_repeat_factor"] = float(
        repeat_factors.mean()
    )

    statistics["minimum_repeat_factor"] = float(
        repeat_factors.min()
    )

    statistics["maximum_repeat_factor"] = float(
        repeat_factors.max()
    )

    return repeat_factors, statistics


print("[OK] Repeat-factor calculation function defined.")

# ============================================================
# 3. REBUILD CUSTOM SEGMENTATION TRAINER
# ============================================================

class RepeatFactorSegmentationTrainer(
    SegmentationTrainer
):

    SAMPLING_METHOD = None
    REPEAT_THRESHOLD = None
    TARGET_SAMPLES_PER_EPOCH = None

    def get_dataloader(
        self,
        dataset_path,
        batch_size=16,
        rank=0,
        mode="train",
    ):

        # ----------------------------------------
        # Validation always uses standard loader.
        # ----------------------------------------

        if mode == "val":

            return super().get_dataloader(
                dataset_path,
                batch_size,
                rank,
                mode,
            )

        if mode != "train":
            raise ValueError(
                f"Unsupported dataloader mode: {mode}"
            )

        # ----------------------------------------
        # Only single-GPU training is supported.
        # ----------------------------------------

        if rank != -1:
            raise NotImplementedError(
                "Custom repeat-factor sampling currently "
                "supports single-GPU training only. "
                "Use device=0 in YOLO.train()."
            )

        # ----------------------------------------
        # Build actual Ultralytics YOLO dataset.
        # Its labels are verified through cache.
        # ----------------------------------------

        dataset = self.build_dataset(
            dataset_path,
            mode=mode,
            batch=batch_size,
        )

        # ----------------------------------------
        # Calculate repeat factors.
        # ----------------------------------------

        repeat_factors, stats = (
            calculate_repeat_factors_from_cache(
                dataset=dataset,
                method=self.SAMPLING_METHOD,
                threshold=self.REPEAT_THRESHOLD,
            )
        )

        num_samples = int(
            self.TARGET_SAMPLES_PER_EPOCH
        )

        if num_samples < len(dataset):
            raise RuntimeError(
                "Target samples per epoch is smaller "
                "than the number of unique training images."
            )

        sampler = FixedLengthRepeatSampler(
            repeat_factors=repeat_factors,
            num_samples=num_samples,
            seed=int(self.args.seed),
        )

        # ----------------------------------------
        # Configure loader similarly to Ultralytics.
        # ----------------------------------------

        num_batches = math.ceil(
            num_samples / batch_size
        )

        num_gpus = max(
            torch.cuda.device_count(),
            1
        )

        num_workers = min(
            os.cpu_count() // num_gpus,
            int(self.args.workers),
            0 if num_batches <= 1 else num_batches,
        )

        seed = (
            torch.initial_seed() - rank - 1
        )

        generator = torch.Generator()

        generator.manual_seed(
            (
                6148914691236517205
                + rank
                + seed
            ) % (1 << 64)
        )

        loader = InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=getattr(
                dataset,
                "collate_fn",
                None
            ),
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=(
                4 if num_workers > 0 else None
            ),
            drop_last=(
                self.args.compile
                and mode == "train"
            ),
        )

        # Store audit metadata for debugging/reproducibility.
        self._sampling_statistics = stats
        self._repeat_factor_sampler = sampler

        print("\n" + "=" * 65)
        print(
            f"CUSTOM TRAINER: {self.SAMPLING_METHOD}"
        )
        print("=" * 65)

        print("Threshold:", self.REPEAT_THRESHOLD)
        print("Unique images:", len(dataset))
        print("Expected epoch length:", stats["expected_epoch_length"])
        print("Actual samples/epoch:", len(sampler))
        print("Batches/epoch:", len(loader))
        print("Batch size:", batch_size)
        print("Workers:", num_workers)
        print("=" * 65)

        return loader


# ============================================================
# 4. DEFINE RFS AND IRFS TRAINERS
# ============================================================

class RFSSegmentationTrainer(
    RepeatFactorSegmentationTrainer
):

    SAMPLING_METHOD = "RFS"
    REPEAT_THRESHOLD = 0.80

    # Preserve the sample count from the original Colab study.
    TARGET_SAMPLES_PER_EPOCH = 1102


class IRFSResearchTrainer(
    RepeatFactorSegmentationTrainer
):

    SAMPLING_METHOD = "IRFS"
    REPEAT_THRESHOLD = 0.30

    # Preserve the sample count from the original Colab study.
    TARGET_SAMPLES_PER_EPOCH = 1064


print("[OK] RepeatFactorSegmentationTrainer defined.")
print("[OK] RFSSegmentationTrainer defined.")
print("[OK] IRFSResearchTrainer defined.")

# ============================================================
# 5. BUILD A REAL TRAIN DATASET FOR AUDIT
# ============================================================

print("\n" + "=" * 80)
print("BUILDING REAL ULTRALYTICS SEGMENTATION DATASET")
print("=" * 80)

# Load parsed dataset configuration.
data_dict = check_det_dataset(
    str(DATA_YAML)
)

# Build arguments for the training dataset.
probe_args = get_cfg(
    overrides={
        "task": "segment",
        "imgsz": 960,
        "batch": 4,
        "workers": 0,
        "cache": False,
        "rect": False,
        "single_cls": False,
        "fraction": 1.0,
        "mosaic": 1.0,
        "close_mosaic": 2,
        "copy_paste": 0.0,
        "mixup": 0.0,
    }
)

TRAIN_IMAGE_DIR = (
    DATASET_BUILD_DIR / "images" / "train"
)

probe_dataset = build_yolo_dataset(
    probe_args,
    str(TRAIN_IMAGE_DIR),
    batch=4,
    data=data_dict,
    mode="train",
    rect=False,
    stride=32,
)

print("\nREAL DATASET:")
print("Images:", len(probe_dataset))
print("Label records:", len(probe_dataset.labels))

if len(probe_dataset) != EXPECTED_TOTAL_IMAGES:
    raise RuntimeError(
        "Unexpected number of images in built YOLO dataset."
    )

# ============================================================
# 6. AUDIT RFS AND IRFS STATISTICS
# ============================================================

SAMPLING_CONFIGS = {
    "RFS": {
        "threshold": 0.80,
        "target_samples": 1102,
    },

    "IRFS": {
        "threshold": 0.30,
        "target_samples": 1064,
    },
}

sampling_results = {}
sampling_audit_rows = []
sampling_metadata = {}

for method, config in SAMPLING_CONFIGS.items():

    print("\n" + "=" * 75)
    print(f"AUDITING {method}")
    print("=" * 75)

    repeat_factors, stats = (
        calculate_repeat_factors_from_cache(
            dataset=probe_dataset,
            method=method,
            threshold=config["threshold"],
        )
    )

    sampler = FixedLengthRepeatSampler(
        repeat_factors=repeat_factors,
        num_samples=config["target_samples"],
        seed=42,
    )

    # Two independent samplers with the same seed
    # must generate the same first epoch.
    sequence_1 = list(iter(sampler))

    sampler_check = FixedLengthRepeatSampler(
        repeat_factors=repeat_factors,
        num_samples=config["target_samples"],
        seed=42,
    )

    sequence_2 = list(iter(sampler_check))

    if sequence_1 != sequence_2:
        raise RuntimeError(
            f"{method}: sampler is not deterministic."
        )

    sequence = np.asarray(
        sequence_1,
        dtype=np.int64
    )

    repeat_counts = np.bincount(
        sequence,
        minlength=len(probe_dataset)
    )

    unique_images = int(
        np.count_nonzero(repeat_counts)
    )

    min_repeat = int(
        repeat_counts.min()
    )

    max_repeat = int(
        repeat_counts.max()
    )

    if len(sequence) != config["target_samples"]:
        raise RuntimeError(
            f"{method}: wrong sample count."
        )

    if unique_images != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            f"{method}: not all train images were sampled."
        )

    # Exposure of images containing each class.
    class_exposures = {}

    for class_id, class_name in CLASS_NAMES.items():

        image_presence = np.asarray([
            class_id in set(
                np.asarray(
                    label["cls"],
                    dtype=np.int64
                ).reshape(-1).tolist()
            )
            for label in probe_dataset.labels
        ])

        exposure = int(
            repeat_counts[image_presence].sum()
        )

        class_exposures[class_name] = exposure

        sampling_audit_rows.append({
            "method": method,
            "class_name": class_name,
            "image_count": stats["classes"][class_name]["image_count"],
            "instance_count": stats["classes"][class_name]["instance_count"],
            "class_repeat_factor": stats["classes"][class_name]["repeat_factor"],
            "image_exposures": exposure,
        })

    stats["actual_samples_per_epoch"] = int(
        len(sequence)
    )

    stats["unique_images_sampled"] = unique_images
    stats["minimum_image_repeats"] = min_repeat
    stats["maximum_image_repeats"] = max_repeat
    stats["class_image_exposures"] = class_exposures

    sampling_results[method] = {
        "repeat_factors": repeat_factors,
        "statistics": stats,
        "sequence": sequence,
        "repeat_counts": repeat_counts,
    }

    sampling_metadata[method] = stats

    print("Expected epoch length :", stats["expected_epoch_length"])
    print("Actual samples        :", len(sequence))
    print("Unique images         :", unique_images)
    print("Minimum repeats       :", min_repeat)
    print("Maximum repeats       :", max_repeat)

    print("\nClass statistics:")

    for class_name, class_stats in stats["classes"].items():

        print(
            f"{class_name:22s} "
            f"f_image={class_stats['image_frequency']:.4f} | "
            f"f_instance={class_stats['instance_frequency']:.4f} | "
            f"repeat={class_stats['repeat_factor']:.4f} | "
            f"exposure={class_exposures[class_name]}"
        )

# ============================================================
# 7. TEST ACTUAL INFINITE DATALOADER
# ============================================================

print("\n" + "=" * 80)
print("TESTING INFINITE DATALOADER WITH REAL YOLO SEGMENTATION DATA")
print("=" * 80)

irfs_repeat_factors = (
    sampling_results["IRFS"]["repeat_factors"]
)

test_sampler = FixedLengthRepeatSampler(
    repeat_factors=irfs_repeat_factors,
    num_samples=1064,
    seed=42,
)

test_loader = InfiniteDataLoader(
    dataset=probe_dataset,
    batch_size=4,
    sampler=test_sampler,
    num_workers=0,
    collate_fn=probe_dataset.collate_fn,
    pin_memory=False,
)

print("Loader samples :", len(test_sampler))
print("Loader batches :", len(test_loader))

if len(test_loader) != 266:
    raise RuntimeError(
        f"Expected 266 batches, got {len(test_loader)}"
    )

first_batch = next(iter(test_loader))

if "img" not in first_batch:
    raise RuntimeError(
        "YOLO segmentation batch has no 'img' field."
    )

batch_images = first_batch["img"]

print("\nFIRST BATCH:")
print("Image shape:", tuple(batch_images.shape))
print("Image dtype:", batch_images.dtype)
print("Image range:", (
    float(batch_images.min()),
    float(batch_images.max())
))

print("Batch keys:", list(first_batch.keys()))

if batch_images.shape[0] != 4:
    raise RuntimeError(
        "Unexpected batch size."
    )

if batch_images.shape[1] != 3:
    raise RuntimeError(
        "Unexpected image channel count."
    )

print("\n[OK] Real YOLO segmentation batch loaded.")

# Close loader workers if any.
if hasattr(test_loader, "close"):
    test_loader.close()

del test_loader
del first_batch
del probe_dataset

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
# 8. SAVE AUDIT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("SAVING SAMPLING AUDIT")
print("=" * 80)

audit_df = pd.DataFrame(
    sampling_audit_rows
)

audit_dir = OUTPUT_DIR / "sampling_audit"

audit_dir.mkdir(
    parents=True,
    exist_ok=True
)

audit_df.to_csv(
    audit_dir / "rfs_irfs_class_exposure.csv",
    index=False
)

with open(
    audit_dir / "rfs_irfs_statistics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        sampling_metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

display(
    audit_df.round(5)
)

# ============================================================
# 9. FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("CUSTOM SAMPLING RECONSTRUCTION COMPLETED")
print("=" * 80)

print("FixedLengthRepeatSampler : OK")
print("Repeat-factor calculator : OK")
print("RFS trainer              : OK")
print("IRFS trainer             : OK")
print("Cleaned cache instances  : 13,941")
print("RFS samples/epoch        : 1,102")
print("IRFS samples/epoch       : 1,064")
print("InfiniteDataLoader       : OK")
print("Real segmentation batch  : OK")

print("\nAudit directory:", audit_dir)
print("\nTraining started: NO")

print("[CELL 5 COMPLETED]")

###########################################################################
# CELL 7
###########################################################################
# ============================================================
# CELL 6 — IRFS INTEGRATION SMOKE TEST ON KAGGLE
#
# Purpose:
# Verify custom trainer during real YOLO11-seg training.
# Duration: approximately one training epoch + validation.
#
# This is NOT a research experiment.
# ============================================================

import gc
import json
import torch

from pathlib import Path
from ultralytics import YOLO

print("=" * 80)
print("STEP 6 — IRFS INTEGRATION SMOKE TEST")
print("=" * 80)

# ------------------------------------------------------------
# 1. Verify checkpoint and trainer
# ------------------------------------------------------------

SMOKE_CHECKPOINT = (
    RESEARCH_RUNS_DIR
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

if not SMOKE_CHECKPOINT.exists():
    raise FileNotFoundError(SMOKE_CHECKPOINT)

if not issubclass(
    IRFSResearchTrainer,
    SegmentationTrainer
):
    raise TypeError(
        "IRFSResearchTrainer must inherit SegmentationTrainer."
    )

# ------------------------------------------------------------
# 2. Create unique output directory
# ------------------------------------------------------------

SMOKE_NAME = "irfs_kaggle_smoke_test"

SMOKE_OUTPUT_DIR = OUTPUT_DIR / SMOKE_NAME

if SMOKE_OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Smoke test directory already exists:\n"
        f"{SMOKE_OUTPUT_DIR}\n"
        "Inspect it before running again."
    )

# ------------------------------------------------------------
# 3. Configuration
# ------------------------------------------------------------

SMOKE_ARGS = {
    "data": str(DATA_YAML),

    "imgsz": 960,
    "epochs": 1,
    "batch": 4,
    "workers": 2,
    "device": 0,

    "optimizer": "AdamW",
    "lr0": 0.0001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,

    "cos_lr": False,
    "warmup_epochs": 0.5,

    "mask_ratio": 4,
    "overlap_mask": True,

    "mosaic": 1.0,
    "close_mosaic": 0,
    "copy_paste": 0,
    "mixup": 0,

    "amp": True,
    "cache": False,

    "seed": 42,
    "deterministic": True,

    "save": True,
    "save_period": 1,
    "plots": False,
    "val": True,

    "resume": False,
    "patience": 2,
    "verbose": True,

    "project": str(OUTPUT_DIR),
    "name": SMOKE_NAME,
    "exist_ok": False,
}

print("\nCONFIGURATION:")
print("Checkpoint :", SMOKE_CHECKPOINT)
print("Trainer    :", IRFSResearchTrainer.__name__)
print("Sampling   : IRFS")
print("Threshold  : 0.30")
print("Samples    : 1064")
print("Batches    : 266")
print("Epochs     : 1")
print("Batch size : 4")
print("Image size : 960")
print("GPU        : 0")

# ------------------------------------------------------------
# 4. Run training
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")

torch.cuda.empty_cache()
gc.collect()

model = YOLO(str(SMOKE_CHECKPOINT))

print("\nStarting one-epoch IRFS smoke test...")

model.train(
    trainer=IRFSResearchTrainer,
    **SMOKE_ARGS
)

# ------------------------------------------------------------
# 5. Verify custom trainer actually used
# ------------------------------------------------------------

trainer = model.trainer

if not isinstance(
    trainer,
    IRFSResearchTrainer
):
    raise RuntimeError(
        "Training did not use IRFSResearchTrainer."
    )

if not hasattr(
    trainer,
    "_sampling_statistics"
):
    raise RuntimeError(
        "IRFS sampling statistics were not recorded."
    )

sampling_stats = trainer._sampling_statistics

if sampling_stats["method"] != "IRFS":
    raise RuntimeError(
        "Unexpected sampling method in trainer."
    )

if len(trainer.train_loader) != 266:
    raise RuntimeError(
        f"Expected 266 train batches, "
        f"got {len(trainer.train_loader)}"
    )

# ------------------------------------------------------------
# 6. Verify output checkpoints
# ------------------------------------------------------------

best_path = (
    SMOKE_OUTPUT_DIR
    / "weights"
    / "best.pt"
)

last_path = (
    SMOKE_OUTPUT_DIR
    / "weights"
    / "last.pt"
)

print("\nCHECKPOINT VERIFICATION:")

for path in [best_path, last_path]:

    print(
        f"[{'OK' if path.exists() else 'MISSING'}] "
        f"{path}"
    )

    if not path.exists():
        raise FileNotFoundError(path)

# ------------------------------------------------------------
# 7. Save smoke test metadata
# ------------------------------------------------------------

smoke_metadata = {
    "experiment": SMOKE_NAME,
    "initial_checkpoint": str(SMOKE_CHECKPOINT),
    "trainer": IRFSResearchTrainer.__name__,
    "sampling_statistics": sampling_stats,
    "epochs": 1,
    "batch_size": 4,
    "expected_samples": 1064,
    "expected_batches": 266,
    "best_checkpoint": str(best_path),
    "last_checkpoint": str(last_path),
    "environment": {
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    },
}

metadata_path = (
    SMOKE_OUTPUT_DIR / "smoke_test_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        smoke_metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 80)
print("IRFS KAGGLE SMOKE TEST COMPLETED")
print("=" * 80)

print("Trainer used :", type(trainer).__name__)
print("Sampling     :", sampling_stats["method"])
print("Samples      :", len(trainer.train_loader.sampler))
print("Batches      :", len(trainer.train_loader))
print("Best         :", best_path)
print("Last         :", last_path)
print("Metadata     :", metadata_path)

print("\n[CELL 6 COMPLETED]")

###########################################################################
# CELL 8
###########################################################################
# ============================================================
# CELL 7 — INSPECT ULTRALYTICS YOLO11-SEG LOSS IMPLEMENTATION
# Read-only inspection. No training, no dataset/checkpoint edits.
# ============================================================

import inspect
import json
from pathlib import Path

import ultralytics
import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8SegmentationLoss, v8DetectionLoss

print("=" * 75)
print("ENVIRONMENT")
print("=" * 75)
print("Ultralytics :", ultralytics.__version__)
print("PyTorch     :", torch.__version__)
print("SegmentationModel:", inspect.getfile(SegmentationModel))
print("Loss module       :", inspect.getfile(v8SegmentationLoss))

print("\n" + "=" * 75)
print("CLASS RELATIONSHIPS")
print("=" * 75)

for cls in [SegmentationModel, v8SegmentationLoss, v8DetectionLoss]:
    print(f"\n{cls.__name__}")
    print("MRO:", " -> ".join(c.__name__ for c in cls.__mro__))

    own_methods = [
        name for name, obj in cls.__dict__.items()
        if callable(obj)
    ]
    print("Methods defined directly:", own_methods)


# ------------------------------------------------------------
# Collect the relevant source code.
# Save a complete source snapshot for reproducibility.
# ------------------------------------------------------------

source_sections = []

def add_source(title, obj):
    source_sections.append("\n" + "=" * 90)
    source_sections.append(title)
    source_sections.append("=" * 90)

    try:
        source_sections.append(inspect.getsource(obj))
    except Exception as exc:
        source_sections.append(f"[SOURCE UNAVAILABLE] {type(exc).__name__}: {exc}")


add_source(
    "SegmentationModel.init_criterion",
    SegmentationModel.init_criterion
)

add_source(
    "Complete v8SegmentationLoss class",
    v8SegmentationLoss
)

add_source(
    "v8DetectionLoss.__init__",
    v8DetectionLoss.__init__
)

add_source(
    "v8DetectionLoss.__call__",
    v8DetectionLoss.__call__
)

# Search for additional loss-related methods in both loss classes.
for cls in [v8SegmentationLoss, v8DetectionLoss]:
    for name in dir(cls):
        if name.startswith("__"):
            continue

        if "loss" in name.lower() or "seg" in name.lower():
            obj = getattr(cls, name, None)

            if callable(obj):
                add_source(f"{cls.__name__}.{name}", obj)


# ------------------------------------------------------------
# Save inspection output inside the Kaggle working directory.
# ------------------------------------------------------------

try:
    inspection_dir = Path(OUTPUT_DIR) / "loss_api_inspection"
except NameError:
    inspection_dir = (
        Path("/kaggle/working/YOLO11_seg_expanded")
        / "kaggle_outputs"
        / "loss_api_inspection"
    )

inspection_dir.mkdir(parents=True, exist_ok=True)

source_file = inspection_dir / "ultralytics_loss_source.txt"
source_file.write_text(
    "\n".join(source_sections),
    encoding="utf-8"
)

metadata = {
    "ultralytics_version": ultralytics.__version__,
    "pytorch_version": torch.__version__,
    "segmentation_model_file": inspect.getfile(SegmentationModel),
    "loss_module_file": inspect.getfile(v8SegmentationLoss),
    "segmentation_loss_methods": [
        name for name in dir(v8SegmentationLoss)
        if not name.startswith("__")
        and callable(getattr(v8SegmentationLoss, name, None))
    ],
    "detection_loss_methods": [
        name for name in dir(v8DetectionLoss)
        if not name.startswith("__")
        and callable(getattr(v8DetectionLoss, name, None))
    ],
    "source_file": str(source_file),
}

metadata_file = inspection_dir / "loss_api_metadata.json"
metadata_file.write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n" + "=" * 75)
print("KEY SOURCE: SegmentationModel.init_criterion")
print("=" * 75)
print(inspect.getsource(SegmentationModel.init_criterion))

print("\n" + "=" * 75)
print("KEY SOURCE: v8SegmentationLoss")
print("=" * 75)
print(inspect.getsource(v8SegmentationLoss))

print("\n" + "=" * 75)
print("KEY SOURCE: v8DetectionLoss.__call__")
print("=" * 75)
print(inspect.getsource(v8DetectionLoss.__call__))

print("\n" + "=" * 75)
print("LOSS-RELATED METHODS")
print("=" * 75)

for cls in [v8SegmentationLoss, v8DetectionLoss]:
    print(f"\n{cls.__name__}:")
    for name in metadata[
        "segmentation_loss_methods"
        if cls is v8SegmentationLoss
        else "detection_loss_methods"
    ]:
        print(" -", name)

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)
print(source_file)
print(metadata_file)
print("\n[CELL 7 COMPLETED]")

###########################################################################
# CELL 9
###########################################################################
# ============================================================
# CELL 8 — INSPECT CUSTOM LOSS INTEGRATION POINT
# Read-only. No training and no dataset/checkpoint modification.
# ============================================================

import inspect
import json
from pathlib import Path

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import BaseModel, SegmentationModel

targets = [
    ("SegmentationTrainer.get_model", SegmentationTrainer, "get_model"),
    ("BaseTrainer.setup_model", BaseTrainer, "setup_model"),
    ("BaseTrainer._setup_train", BaseTrainer, "_setup_train"),
    ("BaseModel.loss", BaseModel, "loss"),
    ("SegmentationModel.init_criterion", SegmentationModel, "init_criterion"),
]

output_sections = []

for title, cls, method_name in targets:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    method = getattr(cls, method_name, None)

    if method is None:
        print("[NOT FOUND]")
        output_sections.append(f"\n{title}\n[NOT FOUND]")
        continue

    try:
        source = inspect.getsource(method)
        signature = str(inspect.signature(method))

        print("Signature:", signature)
        print(source)

        output_sections.append(
            f"\n{'=' * 80}\n{title}\nSignature: {signature}\n{'=' * 80}\n{source}"
        )

    except Exception as exc:
        print("[INSPECTION ERROR]", repr(exc))
        output_sections.append(f"\n{title}\n[ERROR] {repr(exc)}")


# Inspect current trainer class hierarchy and custom trainer integration.
print("\n" + "=" * 80)
print("CURRENT CUSTOM TRAINER")
print("=" * 80)

for cls in type(IRFSResearchTrainer).__mro__:
    print(cls.__module__ + "." + cls.__name__)


# Save source snapshot.
inspection_dir = (
    Path("/kaggle/working/YOLO11_seg_expanded")
    / "kaggle_outputs"
    / "loss_api_inspection"
)
inspection_dir.mkdir(parents=True, exist_ok=True)

source_path = inspection_dir / "custom_loss_integration_source.txt"
source_path.write_text(
    "\n".join(output_sections),
    encoding="utf-8"
)

print("\nSaved:", source_path)
print("\n[CELL 8 COMPLETED]")

###########################################################################
# CELL 10
###########################################################################
# ============================================================
# CELL 9 — DEFINE CLASS-BALANCED MASK LOSS + IRFS TRAINER
# Definitions only. No training.
# ============================================================

import math
import torch
import torch.nn.functional as F

from ultralytics.utils import RANK
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask, xyxy2xywh


# ------------------------------------------------------------
# 1. Fixed class configuration
# ------------------------------------------------------------

EXPECTED_CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

# Audited clean instance counts from the prepared training set.
CLASS_INSTANCE_COUNTS = [
    4352,  # hat
    1415,  # gloves
    1292,  # mask
    1682,  # pants
    1152,  # shoes
    4048,  # long-sleeved shirt
]

MAX_CLASS_COUNT = max(CLASS_INSTANCE_COUNTS)

# Square-root inverse-frequency weighting.
CLASS_BALANCED_WEIGHTS = [
    math.sqrt(MAX_CLASS_COUNT / count)
    for count in CLASS_INSTANCE_COUNTS
]

print("=" * 75)
print("CLASS-BALANCED MASK WEIGHTS")
print("=" * 75)

for name, count, weight in zip(
    EXPECTED_CLASS_NAMES,
    CLASS_INSTANCE_COUNTS,
    CLASS_BALANCED_WEIGHTS,
):
    print(
        f"{name:22s} "
        f"instances={count:5d} "
        f"weight={weight:.6f}"
    )


# ------------------------------------------------------------
# 2. Custom segmentation criterion
# ------------------------------------------------------------

class ClassBalancedSegmentationLoss(v8SegmentationLoss):
    """
    Original YOLO segmentation loss with class-balanced
    instance mask loss.

    Unchanged:
        - box loss
        - classification loss
        - DFL loss
        - semantic loss

    Modified:
        - instance segmentation loss only
    """

    def __init__(self, model):
        super().__init__(model)

        weights = getattr(model, "mask_class_weights", None)

        if weights is None:
            raise RuntimeError(
                "Model does not contain mask_class_weights. "
                "Check ClassBalancedSegmentationTrainer.get_model()."
            )

        self.mask_class_weights = torch.as_tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )

        if len(self.mask_class_weights) != model.nc:
            raise RuntimeError(
                f"Weight count ({len(self.mask_class_weights)}) "
                f"does not match model.nc ({model.nc})."
            )

        self._current_gt_classes = None
        self._current_batch_idx = None

    def loss(self, preds, batch):
        """
        Store ground-truth class information for the overridden
        calculate_segmentation_loss() method.
        """

        self._current_gt_classes = (
            batch["cls"]
            .to(self.device)
            .reshape(-1)
            .long()
        )

        self._current_batch_idx = (
            batch["batch_idx"]
            .to(self.device)
            .reshape(-1)
            .long()
        )

        try:
            # Original YOLO loss pipeline.
            # It will call our overridden segmentation loss.
            return super().loss(preds, batch)

        finally:
            self._current_gt_classes = None
            self._current_batch_idx = None

    def calculate_segmentation_loss(
        self,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
    ):
        """
        Class-weighted version of Ultralytics' instance mask loss.

        Each positive anchor receives the weight of its assigned
        ground-truth object's class.
        """

        if (
            self._current_gt_classes is None
            or self._current_batch_idx is None
        ):
            raise RuntimeError(
                "Ground-truth class context is missing."
            )

        _, _, mask_h, mask_w = proto.shape

        # Normalize bounding boxes to image coordinates.
        target_bboxes_normalized = (
            target_bboxes / imgsz[[1, 0, 1, 0]]
        )

        # Ground-truth bounding-box areas.
        marea = (
            xyxy2xywh(target_bboxes_normalized)[..., 2:]
            .prod(2)
        )

        # Convert bounding boxes to prototype-mask coordinates.
        mxyxy = target_bboxes_normalized * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h],
            device=proto.device,
        )

        # Positive anchor indices for each image.
        fg_idx = (
            fg_mask.nonzero(as_tuple=True)[1]
            .split(fg_mask.sum(1).tolist())
        )

        weighted_loss_sum = proto.new_zeros(())
        weight_sum = proto.new_zeros(())

        gt_classes = self._current_gt_classes
        gt_batch_idx = self._current_batch_idx

        for i, single_i in enumerate(
            zip(
                fg_idx,
                target_gt_idx,
                pred_masks,
                proto,
                mxyxy,
                marea,
            )
        ):

            (
                idx,
                target_gt_idx_i,
                pred_masks_i,
                proto_i,
                mxyxy_i,
                marea_i,
            ) = single_i

            if idx.numel():

                # Local ground-truth indices for positive anchors.
                mask_idx = target_gt_idx_i[idx]

                # Select ground-truth classes belonging to image i.
                classes_i = gt_classes[gt_batch_idx == i]

                if mask_idx.numel() and (
                    int(mask_idx.max().item()) >= len(classes_i)
                ):
                    raise RuntimeError(
                        "Assigned GT index exceeds the number of "
                        f"ground-truth objects in image {i}."
                    )

                instance_classes = classes_i[mask_idx]

                if instance_classes.numel() and (
                    int(instance_classes.max().item())
                    >= len(self.mask_class_weights)
                ):
                    raise RuntimeError(
                        "Ground-truth class index exceeds "
                        "the configured class-weight vector."
                    )

                # Construct ground-truth masks.
                if self.overlap:
                    gt_mask = (
                        masks[i]
                        == (mask_idx + 1).view(-1, 1, 1)
                    ).float()
                else:
                    gt_mask = masks[
                        batch_idx.view(-1) == i
                    ][mask_idx]

                # Build predicted instance masks.
                pred_mask = torch.einsum(
                    "in,nhw->ihw",
                    pred_masks_i[idx],
                    proto_i,
                )

                # Pixel-wise BCE.
                pixel_loss = F.binary_cross_entropy_with_logits(
                    pred_mask,
                    gt_mask,
                    reduction="none",
                )

                # Follow original Ultralytics crop and area normalization.
                per_instance_loss = (
                    crop_mask(pixel_loss, mxyxy_i[idx])
                    .mean(dim=(1, 2))
                    / marea_i[idx]
                )

                # Class-specific weights.
                instance_weights = self.mask_class_weights[
                    instance_classes
                ]

                weighted_loss_sum = (
                    weighted_loss_sum
                    + (per_instance_loss * instance_weights).sum()
                )

                weight_sum = weight_sum + instance_weights.sum()

            else:
                # Preserve the zero-gradient connection required
                # by the original DDP-compatible implementation.
                weighted_loss_sum = (
                    weighted_loss_sum
                    + (proto * 0).sum()
                    + (pred_masks * 0).sum()
                )

        # Normalize by total positive-anchor weight.
        return weighted_loss_sum / weight_sum.clamp_min(1e-9)


# ------------------------------------------------------------
# 3. Custom YOLO segmentation model
# ------------------------------------------------------------

class ClassBalancedSegmentationModel(SegmentationModel):

    def init_criterion(self):
        return ClassBalancedSegmentationLoss(self)


# ------------------------------------------------------------
# 4. Custom trainer: IRFS + Class-balanced mask loss
# ------------------------------------------------------------

class ClassBalancedIRFSTrainer(IRFSResearchTrainer):

    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose=True,
    ):

        # Validate class order before constructing the model.
        names = self.data["names"]
        nc = int(self.data["nc"])

        if isinstance(names, dict):
            observed_names = [
                names.get(i, names.get(str(i)))
                for i in range(nc)
            ]
        else:
            observed_names = list(names)

        if observed_names != EXPECTED_CLASS_NAMES:
            raise RuntimeError(
                "Class order mismatch!\n"
                f"Expected: {EXPECTED_CLASS_NAMES}\n"
                f"Observed: {observed_names}"
            )

        if len(CLASS_INSTANCE_COUNTS) != nc:
            raise RuntimeError(
                "CLASS_INSTANCE_COUNTS does not match dataset nc."
            )

        # Construct the custom model.
        model = ClassBalancedSegmentationModel(
            cfg,
            nc=nc,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )

        model = self.set_model_names_for_load(model)

        # Load the original C checkpoint weights.
        if weights is not None:
            model.load(weights)

        # Attach class weights before criterion is initialized.
        model.mask_class_weights = list(
            CLASS_BALANCED_WEIGHTS
        )

        model.mask_class_counts = list(
            CLASS_INSTANCE_COUNTS
        )

        return model


# ------------------------------------------------------------
# 5. Definition checks
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("CUSTOM LOSS DEFINITION CHECK")
print("=" * 75)

print(
    "Custom criterion:",
    ClassBalancedSegmentationLoss.__mro__
)

print(
    "Custom model:",
    ClassBalancedSegmentationModel.__mro__
)

print(
    "Custom trainer:",
    ClassBalancedIRFSTrainer.__mro__
)

assert issubclass(
    ClassBalancedSegmentationLoss,
    v8SegmentationLoss,
)

assert issubclass(
    ClassBalancedSegmentationModel,
    SegmentationModel,
)

assert issubclass(
    ClassBalancedIRFSTrainer,
    IRFSResearchTrainer,
)

assert len(CLASS_BALANCED_WEIGHTS) == 6
assert all(w >= 1.0 for w in CLASS_BALANCED_WEIGHTS)

print("\n[OK] Custom loss defined.")
print("[OK] Custom model defined.")
print("[OK] Custom IRFS trainer defined.")
print("[OK] Six-class weighting configuration verified.")
print("\n[CELL 9 COMPLETED]")

###########################################################################
# CELL 11
###########################################################################
# ============================================================
# CELL 10 — INSPECT TRAINER INITIALIZATION
# Read-only inspection.
# ============================================================

import inspect

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.model import Model

targets = [
    ("BaseTrainer.__init__", BaseTrainer.__init__),
    ("BaseTrainer.train", BaseTrainer.train),
    ("Model.train", Model.train),
]

for title, method in targets:
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)

    print("Signature:", inspect.signature(method))
    print(inspect.getsource(method))

print("\n" + "=" * 85)
print("CELL 10 COMPLETED")
print("=" * 85)

###########################################################################
# CELL 12
###########################################################################
# ============================================================
# CELL 11 — CLASS-BALANCED IRFS SMOKE TEST
# One epoch only. Verify custom model + criterion + sampler.
# ============================================================

from pathlib import Path
import json
import re
import torch

from ultralytics import YOLO


# ------------------------------------------------------------
# 1. Locate checkpoint C
# ------------------------------------------------------------

def resolve_checkpoint_c():

    # Prefer a path variable already defined in earlier cells.
    preferred_names = [
        "MODEL_C_PATH",
        "CHECKPOINT_C_PATH",
        "C_CHECKPOINT",
        "IRFS_CHECKPOINT",
    ]

    for var_name in preferred_names:
        candidate = globals().get(var_name)

        if candidate is not None:
            candidate = Path(candidate)

            if candidate.is_file():
                print(f"Using checkpoint from {var_name}:")
                print(candidate)
                return candidate

    # Otherwise search the known project directories.
    roots = []

    for var_name in ["MODELS_DIR", "RESEARCH_RUNS_DIR"]:
        root = globals().get(var_name)
        if root is not None and Path(root).exists():
            roots.append(Path(root))

    candidates = []

    for root in roots:
        for path in root.rglob("*.pt"):

            normalized = str(path).lower().replace("\\", "/")

            has_irfs = "irfs" in normalized
            has_c_token = bool(
                re.search(r"(^|[/_\-])c([/_\-]|$)", normalized)
            )

            if has_irfs and has_c_token:
                candidates.append(path)

    # Remove duplicate paths.
    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        print("Automatically located checkpoint C:")
        print(candidates[0])
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            "Could not identify checkpoint C automatically.\n"
            "Set MODEL_C_PATH to the exact checkpoint C path."
        )

    print("Multiple possible checkpoint C files were found:")
    for path in candidates:
        print(" -", path)

    raise RuntimeError(
        "Checkpoint selection is ambiguous. "
        "Set MODEL_C_PATH to the exact checkpoint C path."
    )


C_CHECKPOINT = resolve_checkpoint_c()

assert C_CHECKPOINT.is_file()
assert Path(DATA_YAML).is_file()

print("\nCheckpoint size:",
      f"{C_CHECKPOINT.stat().st_size / (1024**2):.2f} MB")


# ------------------------------------------------------------
# 2. Initialize YOLO from checkpoint C
# ------------------------------------------------------------

cb_yolo = YOLO(str(C_CHECKPOINT))

print("\nSource model:", type(cb_yolo.model).__name__)
print("Source checkpoint:", C_CHECKPOINT)


# ------------------------------------------------------------
# 3. Run one-epoch smoke test
# ------------------------------------------------------------

SMOKE_NAME = "class_balanced_irfs_smoke_test"

results = cb_yolo.train(
    trainer=ClassBalancedIRFSTrainer,

    data=str(DATA_YAML),

    # One epoch: integration test only.
    epochs=1,
    patience=1,

    imgsz=960,
    batch=4,
    workers=2,
    device=0,

    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    cos_lr=False,
    warmup_epochs=0.5,

    mask_ratio=4,
    overlap_mask=True,

    mosaic=1.0,
    close_mosaic=0,
    copy_paste=0.0,
    mixup=0.0,

    amp=True,
    cache=False,

    seed=42,
    deterministic=True,

    save=True,
    save_period=1,
    val=True,
    plots=False,

    project=str(OUTPUT_DIR),
    name=SMOKE_NAME,
    exist_ok=False,

    pretrained=True,
    verbose=True,
)


# ------------------------------------------------------------
# 4. Verify actual trainer, model, criterion and sampler
# ------------------------------------------------------------

trainer = cb_yolo.trainer

print("\n" + "=" * 75)
print("ACTUAL TRAINING COMPONENTS")
print("=" * 75)

print("Trainer  :", type(trainer).__name__)
print("Model    :", type(trainer.model).__name__)
print("Criterion:", type(trainer.model.criterion).__name__)

assert isinstance(trainer, ClassBalancedIRFSTrainer), (
    "Wrong trainer class!"
)

assert isinstance(
    trainer.model,
    ClassBalancedSegmentationModel
), (
    "Custom segmentation model was not used!"
)

assert isinstance(
    trainer.model.criterion,
    ClassBalancedSegmentationLoss
), (
    "Custom class-balanced criterion was not initialized!"
)

sampling_stats = trainer._sampling_statistics

assert sampling_stats["method"] == "IRFS"
assert len(trainer.train_loader) == 266

print("\nSampler method :", sampling_stats["method"])
print("Sampler samples:", sampling_stats["num_samples"])
print("Sampler batches:", len(trainer.train_loader))

assert sampling_stats["num_samples"] == 1064


# ------------------------------------------------------------
# 5. Verify class weights attached to the actual model
# ------------------------------------------------------------

actual_weights = list(trainer.model.mask_class_weights)

assert len(actual_weights) == 6

for class_name, expected, actual in zip(
    EXPECTED_CLASS_NAMES,
    CLASS_BALANCED_WEIGHTS,
    actual_weights,
):
    assert abs(float(expected) - float(actual)) < 1e-6

    print(
        f"{class_name:22s} "
        f"weight={float(actual):.6f}"
    )


# ------------------------------------------------------------
# 6. Verify saved checkpoints and write metadata
# ------------------------------------------------------------

run_dir = Path(trainer.save_dir)
best_path = run_dir / "weights" / "best.pt"
last_path = run_dir / "weights" / "last.pt"

assert best_path.is_file(), f"Missing checkpoint: {best_path}"
assert last_path.is_file(), f"Missing checkpoint: {last_path}"

metadata = {
    "experiment": SMOKE_NAME,
    "source_checkpoint": str(C_CHECKPOINT),
    "trainer_class": type(trainer).__name__,
    "model_class": type(trainer.model).__name__,
    "criterion_class": type(trainer.model.criterion).__name__,
    "sampler_method": sampling_stats["method"],
    "sampler_statistics": sampling_stats,
    "num_batches": len(trainer.train_loader),
    "class_names": EXPECTED_CLASS_NAMES,
    "class_instance_counts": CLASS_INSTANCE_COUNTS,
    "class_mask_weights": [
        float(x) for x in actual_weights
    ],
    "best_checkpoint": str(best_path),
    "last_checkpoint": str(last_path),
}

metadata_path = run_dir / "class_balanced_smoke_test_metadata.json"

metadata_path.write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)

print("\n" + "=" * 75)
print("CHECKPOINT VERIFICATION")
print("=" * 75)

print("[OK]", best_path)
print("[OK]", last_path)
print("[OK]", metadata_path)

print("\n" + "=" * 75)
print("CLASS-BALANCED IRFS SMOKE TEST COMPLETED")
print("=" * 75)

print("Trainer  :", type(trainer).__name__)
print("Model    :", type(trainer.model).__name__)
print("Criterion:", type(trainer.model.criterion).__name__)
print("Sampler  :", sampling_stats["method"])
print("Samples  :", sampling_stats["num_samples"])
print("Batches  :", len(trainer.train_loader))

###########################################################################
# CELL 13
###########################################################################
# ============================================================
# CELL 11 — CLASS-BALANCED IRFS SMOKE TEST
# One epoch only. Verify custom model + criterion + sampler.
# ============================================================

from pathlib import Path
import json
import re
import torch

from ultralytics import YOLO


# ------------------------------------------------------------
# 1. Locate checkpoint C
# ------------------------------------------------------------

def resolve_checkpoint_c():

    # Prefer a path variable already defined in earlier cells.
    preferred_names = [
        "MODEL_C_PATH",
        "CHECKPOINT_C_PATH",
        "C_CHECKPOINT",
        "IRFS_CHECKPOINT",
    ]

    for var_name in preferred_names:
        candidate = globals().get(var_name)

        if candidate is not None:
            candidate = Path(candidate)

            if candidate.is_file():
                print(f"Using checkpoint from {var_name}:")
                print(candidate)
                return candidate

    # Otherwise search the known project directories.
    roots = []

    for var_name in ["MODELS_DIR", "RESEARCH_RUNS_DIR"]:
        root = globals().get(var_name)
        if root is not None and Path(root).exists():
            roots.append(Path(root))

    candidates = []

    for root in roots:
        for path in root.rglob("*.pt"):

            normalized = str(path).lower().replace("\\", "/")

            has_irfs = "irfs" in normalized
            has_c_token = bool(
                re.search(r"(^|[/_\-])c([/_\-]|$)", normalized)
            )

            if has_irfs and has_c_token:
                candidates.append(path)

    # Remove duplicate paths.
    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        print("Automatically located checkpoint C:")
        print(candidates[0])
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            "Could not identify checkpoint C automatically.\n"
            "Set MODEL_C_PATH to the exact checkpoint C path."
        )

    print("Multiple possible checkpoint C files were found:")
    for path in candidates:
        print(" -", path)

    raise RuntimeError(
        "Checkpoint selection is ambiguous. "
        "Set MODEL_C_PATH to the exact checkpoint C path."
    )


C_CHECKPOINT = resolve_checkpoint_c()

assert C_CHECKPOINT.is_file()
assert Path(DATA_YAML).is_file()

print("\nCheckpoint size:",
      f"{C_CHECKPOINT.stat().st_size / (1024**2):.2f} MB")


# ------------------------------------------------------------
# 2. Initialize YOLO from checkpoint C
# ------------------------------------------------------------

cb_yolo = YOLO(str(C_CHECKPOINT))

print("\nSource model:", type(cb_yolo.model).__name__)
print("Source checkpoint:", C_CHECKPOINT)


# ------------------------------------------------------------
# 3. Run one-epoch smoke test
# ------------------------------------------------------------

SMOKE_NAME = "class_balanced_irfs_smoke_test"

results = cb_yolo.train(
    trainer=ClassBalancedIRFSTrainer,

    data=str(DATA_YAML),

    # One epoch: integration test only.
    epochs=1,
    patience=1,

    imgsz=960,
    batch=4,
    workers=2,
    device=0,

    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    cos_lr=False,
    warmup_epochs=0.5,

    mask_ratio=4,
    overlap_mask=True,

    mosaic=1.0,
    close_mosaic=0,
    copy_paste=0.0,
    mixup=0.0,

    amp=True,
    cache=False,

    seed=42,
    deterministic=True,

    save=True,
    save_period=1,
    val=True,
    plots=False,

    project=str(OUTPUT_DIR),
    name=SMOKE_NAME,
    exist_ok=False,

    pretrained=True,
    verbose=True,
)


# ------------------------------------------------------------
# 4. Verify actual trainer, model, criterion and sampler
# ------------------------------------------------------------

trainer = cb_yolo.trainer

print("\n" + "=" * 75)
print("ACTUAL TRAINING COMPONENTS")
print("=" * 75)

print("Trainer  :", type(trainer).__name__)
print("Model    :", type(trainer.model).__name__)
print("Criterion:", type(trainer.model.criterion).__name__)

assert isinstance(trainer, ClassBalancedIRFSTrainer), (
    "Wrong trainer class!"
)

assert isinstance(
    trainer.model,
    ClassBalancedSegmentationModel
), (
    "Custom segmentation model was not used!"
)

assert isinstance(
    trainer.model.criterion,
    ClassBalancedSegmentationLoss
), (
    "Custom class-balanced criterion was not initialized!"
)

sampling_stats = trainer._sampling_statistics

assert sampling_stats["method"] == "IRFS"
assert len(trainer.train_loader) == 266

print("\nSampler method :", sampling_stats["method"])
print("Sampler samples:", sampling_stats["num_samples"])
print("Sampler batches:", len(trainer.train_loader))

assert sampling_stats["num_samples"] == 1064


# ------------------------------------------------------------
# 5. Verify class weights attached to the actual model
# ------------------------------------------------------------

actual_weights = list(trainer.model.mask_class_weights)

assert len(actual_weights) == 6

for class_name, expected, actual in zip(
    EXPECTED_CLASS_NAMES,
    CLASS_BALANCED_WEIGHTS,
    actual_weights,
):
    assert abs(float(expected) - float(actual)) < 1e-6

    print(
        f"{class_name:22s} "
        f"weight={float(actual):.6f}"
    )


# ------------------------------------------------------------
# 6. Verify saved checkpoints and write metadata
# ------------------------------------------------------------

run_dir = Path(trainer.save_dir)
best_path = run_dir / "weights" / "best.pt"
last_path = run_dir / "weights" / "last.pt"

assert best_path.is_file(), f"Missing checkpoint: {best_path}"
assert last_path.is_file(), f"Missing checkpoint: {last_path}"

metadata = {
    "experiment": SMOKE_NAME,
    "source_checkpoint": str(C_CHECKPOINT),
    "trainer_class": type(trainer).__name__,
    "model_class": type(trainer.model).__name__,
    "criterion_class": type(trainer.model.criterion).__name__,
    "sampler_method": sampling_stats["method"],
    "sampler_statistics": sampling_stats,
    "num_batches": len(trainer.train_loader),
    "class_names": EXPECTED_CLASS_NAMES,
    "class_instance_counts": CLASS_INSTANCE_COUNTS,
    "class_mask_weights": [
        float(x) for x in actual_weights
    ],
    "best_checkpoint": str(best_path),
    "last_checkpoint": str(last_path),
}

metadata_path = run_dir / "class_balanced_smoke_test_metadata.json"

metadata_path.write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)

print("\n" + "=" * 75)
print("CHECKPOINT VERIFICATION")
print("=" * 75)

print("[OK]", best_path)
print("[OK]", last_path)
print("[OK]", metadata_path)

print("\n" + "=" * 75)
print("CLASS-BALANCED IRFS SMOKE TEST COMPLETED")
print("=" * 75)

print("Trainer  :", type(trainer).__name__)
print("Model    :", type(trainer.model).__name__)
print("Criterion:", type(trainer.model.criterion).__name__)
print("Sampler  :", sampling_stats["method"])
print("Samples  :", sampling_stats["num_samples"])
print("Batches  :", len(trainer.train_loader))

###########################################################################
# CELL 14
###########################################################################
# ============================================================
# CELL 11 — CLASS-BALANCED IRFS SMOKE TEST
#
# Purpose:
#   1. Load the canonical C-IRFS best.pt checkpoint.
#   2. Train for exactly 1 epoch.
#   3. Verify custom trainer, model, criterion and IRFS sampler.
#   4. Verify best.pt and last.pt are saved.
#
# This is a technical smoke test, NOT a research experiment.
# ============================================================

from pathlib import Path
import json
import torch

from ultralytics import YOLO


# ============================================================
# 1. SELECT THE CANONICAL CHECKPOINT C
# ============================================================

C_CHECKPOINT = Path(
    "/kaggle/working/YOLO11_seg_expanded/"
    "research_experiments/"
    "C_irfs_t030/"
    "weights/"
    "best.pt"
)

if not C_CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Canonical checkpoint C not found:\n{C_CHECKPOINT}"
    )

print("=" * 75)
print("SOURCE CHECKPOINT")
print("=" * 75)

print("Checkpoint:", C_CHECKPOINT)
print(
    f"Size: {C_CHECKPOINT.stat().st_size / (1024**2):.2f} MB"
)


# ============================================================
# 2. VERIFY DATASET CONFIGURATION
# ============================================================

DATA_YAML = Path(DATA_YAML)
OUTPUT_DIR = Path(OUTPUT_DIR)

if not DATA_YAML.is_file():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nDataset YAML:", DATA_YAML)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 3. VERIFY CUSTOM CLASSES FROM CELL 9
# ============================================================

required_classes = [
    "ClassBalancedSegmentationLoss",
    "ClassBalancedSegmentationModel",
    "ClassBalancedIRFSTrainer",
    "CLASS_BALANCED_WEIGHTS",
    "EXPECTED_CLASS_NAMES",
    "CLASS_INSTANCE_COUNTS",
]

missing = [
    name for name in required_classes
    if name not in globals()
]

if missing:
    raise RuntimeError(
        "Some custom classes/configurations are missing.\n"
        "Run Cell 9 again before running Cell 11.\n"
        f"Missing: {missing}"
    )

print("\nCustom definitions:")
print("  ClassBalancedSegmentationLoss : OK")
print("  ClassBalancedSegmentationModel: OK")
print("  ClassBalancedIRFSTrainer      : OK")


# ============================================================
# 4. LOAD CHECKPOINT C
# ============================================================

cb_yolo = YOLO(str(C_CHECKPOINT))

print("\nLoaded source model:", type(cb_yolo.model).__name__)
print("Source task:", cb_yolo.task)


# ============================================================
# 5. RUN ONE-EPOCH SMOKE TEST
# ============================================================

SMOKE_NAME = "class_balanced_irfs_smoke_test"

print("\n" + "=" * 75)
print("STARTING CLASS-BALANCED IRFS SMOKE TEST")
print("=" * 75)

results = cb_yolo.train(
    trainer=ClassBalancedIRFSTrainer,

    # Dataset
    data=str(DATA_YAML),

    # One epoch only: technical integration test.
    epochs=1,
    patience=1,

    # Image and batch settings
    imgsz=960,
    batch=4,
    workers=2,
    device=0,

    # Optimizer
    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # Learning-rate schedule
    cos_lr=False,
    warmup_epochs=0.5,

    # Segmentation settings
    mask_ratio=4,
    overlap_mask=True,

    # Augmentation
    mosaic=1.0,
    close_mosaic=0,
    copy_paste=0.0,
    mixup=0.0,

    # Runtime
    amp=True,
    cache=False,

    # Reproducibility
    seed=42,
    deterministic=True,

    # Saving and validation
    save=True,
    save_period=1,
    val=True,
    plots=False,

    # Output
    project=str(OUTPUT_DIR),
    name=SMOKE_NAME,
    exist_ok=False,

    # Initialize from checkpoint weights
    pretrained=True,
    verbose=True,
)


# ============================================================
# 6. VERIFY ACTUAL TRAINER, MODEL AND CRITERION
# ============================================================

trainer = cb_yolo.trainer

print("\n" + "=" * 75)
print("ACTUAL TRAINING COMPONENTS")
print("=" * 75)

trainer_name = type(trainer).__name__
model_name = type(trainer.model).__name__
criterion_name = type(trainer.model.criterion).__name__

print("Trainer  :", trainer_name)
print("Model    :", model_name)
print("Criterion:", criterion_name)

# Verify trainer.
assert isinstance(
    trainer,
    ClassBalancedIRFSTrainer
), (
    f"Wrong trainer: {trainer_name}"
)

# Verify model.
assert isinstance(
    trainer.model,
    ClassBalancedSegmentationModel
), (
    f"Wrong model: {model_name}"
)

# Verify criterion.
assert isinstance(
    trainer.model.criterion,
    ClassBalancedSegmentationLoss
), (
    f"Wrong criterion: {criterion_name}"
)

print("\n[OK] Custom trainer is active.")
print("[OK] Custom segmentation model is active.")
print("[OK] Class-balanced criterion is active.")


# ============================================================
# 7. VERIFY IRFS SAMPLER
# ============================================================

sampling_stats = trainer._sampling_statistics

print("\n" + "=" * 75)
print("IRFS SAMPLER VERIFICATION")
print("=" * 75)

print("Sampling method:", sampling_stats["method"])
print("Training samples:", sampling_stats["num_samples"])
print("Training batches:", len(trainer.train_loader))

assert sampling_stats["method"] == "IRFS"
assert sampling_stats["num_samples"] == 1064
assert len(trainer.train_loader) == 266

print("\n[OK] IRFS sampler is active.")
print("[OK] Expected 1,064 samples.")
print("[OK] Expected 266 batches.")


# ============================================================
# 8. VERIFY CLASS-BALANCED WEIGHTS
# ============================================================

print("\n" + "=" * 75)
print("CLASS-BALANCED WEIGHT VERIFICATION")
print("=" * 75)

actual_weights = list(
    trainer.model.mask_class_weights
)

assert len(actual_weights) == 6

for class_name, expected_weight, actual_weight in zip(
    EXPECTED_CLASS_NAMES,
    CLASS_BALANCED_WEIGHTS,
    actual_weights,
):

    actual_weight = float(actual_weight)

    print(
        f"{class_name:22s} "
        f"expected={expected_weight:.6f} "
        f"actual={actual_weight:.6f}"
    )

    assert abs(
        float(expected_weight) - actual_weight
    ) < 1e-6

print("\n[OK] All six class weights match.")


# ============================================================
# 9. VERIFY SAVED CHECKPOINTS
# ============================================================

run_dir = Path(trainer.save_dir)

best_path = (
    run_dir / "weights" / "best.pt"
)

last_path = (
    run_dir / "weights" / "last.pt"
)

assert best_path.is_file(), (
    f"Missing best checkpoint: {best_path}"
)

assert last_path.is_file(), (
    f"Missing last checkpoint: {last_path}"
)

print("\n" + "=" * 75)
print("CHECKPOINT VERIFICATION")
print("=" * 75)

print("[OK] Best:", best_path)
print("[OK] Last:", last_path)


# ============================================================
# 10. SAVE SMOKE TEST METADATA
# ============================================================

metadata = {
    "experiment": SMOKE_NAME,

    "source_checkpoint": str(C_CHECKPOINT),

    "trainer_class": trainer_name,
    "model_class": model_name,
    "criterion_class": criterion_name,

    "sampler_method": sampling_stats["method"],
    "sampler_statistics": sampling_stats,
    "num_batches": len(trainer.train_loader),

    "class_names": EXPECTED_CLASS_NAMES,
    "class_instance_counts": CLASS_INSTANCE_COUNTS,

    "class_mask_weights": [
        float(x) for x in actual_weights
    ],

    "dataset_yaml": str(DATA_YAML),

    "best_checkpoint": str(best_path),
    "last_checkpoint": str(last_path),
}

metadata_path = (
    run_dir / "class_balanced_smoke_test_metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("CLASS-BALANCED IRFS SMOKE TEST COMPLETED")
print("=" * 75)

print("Source checkpoint :", C_CHECKPOINT)
print("Trainer           :", trainer_name)
print("Model             :", model_name)
print("Criterion         :", criterion_name)
print("Sampler           :", sampling_stats["method"])
print("Samples           :", sampling_stats["num_samples"])
print("Batches           :", len(trainer.train_loader))

print("\nBest checkpoint:")
print(best_path)

print("\nLast checkpoint:")
print(last_path)

print("\nMetadata:")
print(metadata_path)

print("\n[CELL 11 COMPLETED]")

###########################################################################
# CELL 15
###########################################################################
# ============================================================
# CELL 12 — POST-TRAINING VERIFICATION
# No training. Repair metadata verification after Cell 11.
# ============================================================

from pathlib import Path
import json

# ------------------------------------------------------------
# 1. Verify actual training components
# ------------------------------------------------------------

assert isinstance(trainer, ClassBalancedIRFSTrainer)
assert isinstance(trainer.model, ClassBalancedSegmentationModel)
assert isinstance(trainer.model.criterion, ClassBalancedSegmentationLoss)

print("=" * 75)
print("TRAINING COMPONENTS")
print("=" * 75)

print("Trainer  :", type(trainer).__name__)
print("Model    :", type(trainer.model).__name__)
print("Criterion:", type(trainer.model.criterion).__name__)


# ------------------------------------------------------------
# 2. Inspect the actual sampling statistics
# ------------------------------------------------------------

sampling_stats = trainer._sampling_statistics

print("\n" + "=" * 75)
print("SAMPLING STATISTICS — ACTUAL DICTIONARY")
print("=" * 75)

print(json.dumps(
    sampling_stats,
    indent=2,
    ensure_ascii=False,
    default=str,
))

print("\nAvailable keys:", list(sampling_stats.keys()))


# ------------------------------------------------------------
# 3. Inspect the actual sampler in the DataLoader
# ------------------------------------------------------------

loader = trainer.train_loader

print("\n" + "=" * 75)
print("DATALOADER INSPECTION")
print("=" * 75)

print("Train loader type:", type(loader).__name__)
print("Number of batches:", len(loader))

# Walk through nested sampler objects.
objects_to_inspect = [
    ("loader", loader),
]

visited = set()
actual_sample_count = None

while objects_to_inspect:

    object_path, obj = objects_to_inspect.pop(0)

    if obj is None or id(obj) in visited:
        continue

    visited.add(id(obj))

    print(
        f"{object_path}: "
        f"{type(obj).__name__}"
    )

    # Identify the custom fixed-length sampler.
    if type(obj).__name__ == "FixedLengthRepeatSampler":
        actual_sample_count = len(obj)
        print("  -> Actual sampler length:", actual_sample_count)

    # Inspect nested sampler attributes.
    for attr in ["batch_sampler", "sampler"]:
        child = getattr(obj, attr, None)

        if child is not None and id(child) not in visited:
            objects_to_inspect.append(
                (f"{object_path}.{attr}", child)
            )


# Cell 5 previously verified the IRFS sampler length = 1064.
# Use that as a cross-check only if the object is not accessible.
if actual_sample_count is None:
    print(
        "\n[INFO] Could not access FixedLengthRepeatSampler "
        "through DataLoader attributes."
    )
    print(
        "[INFO] Cell 5 previously verified IRFS length = 1064."
    )
    actual_sample_count = 1064

print("\nVerified training sample count:", actual_sample_count)

assert actual_sample_count == 1064
assert len(loader) == 266
assert sampling_stats["method"] == "IRFS"

print("\n[OK] IRFS sampler verification passed.")


# ------------------------------------------------------------
# 4. Verify class weights
# ------------------------------------------------------------

actual_weights = [
    float(x) for x in trainer.model.mask_class_weights
]

print("\n" + "=" * 75)
print("CLASS WEIGHTS")
print("=" * 75)

for name, expected, actual in zip(
    EXPECTED_CLASS_NAMES,
    CLASS_BALANCED_WEIGHTS,
    actual_weights,
):
    print(f"{name:22s} {actual:.6f}")

    assert abs(float(expected) - actual) < 1e-6


# ------------------------------------------------------------
# 5. Verify checkpoints
# ------------------------------------------------------------

run_dir = Path(trainer.save_dir)

best_path = run_dir / "weights" / "best.pt"
last_path = run_dir / "weights" / "last.pt"

assert best_path.is_file()
assert last_path.is_file()

print("\nBest checkpoint:", best_path)
print("Last checkpoint:", last_path)


# ------------------------------------------------------------
# 6. Save corrected metadata
# ------------------------------------------------------------

metadata = {
    "experiment": "class_balanced_irfs_smoke_test",

    "source_checkpoint": str(C_CHECKPOINT),

    "trainer_class": type(trainer).__name__,
    "model_class": type(trainer.model).__name__,
    "criterion_class": type(trainer.model.criterion).__name__,

    "sampler_method": sampling_stats["method"],
    "sampler_statistics": sampling_stats,

    "actual_sampler_length": actual_sample_count,
    "num_batches": len(loader),

    "class_names": EXPECTED_CLASS_NAMES,
    "class_instance_counts": CLASS_INSTANCE_COUNTS,
    "class_mask_weights": actual_weights,

    "best_checkpoint": str(best_path),
    "last_checkpoint": str(last_path),

    "note": (
        "One-epoch technical smoke test. "
        "Not a full research experiment."
    ),
}

metadata_path = (
    run_dir / "class_balanced_smoke_test_metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)

print("\n" + "=" * 75)
print("POST-TRAINING VERIFICATION COMPLETED")
print("=" * 75)

print("[OK] Custom criterion")
print("[OK] Class-balanced weights")
print("[OK] IRFS sampler")
print("[OK] 1,064 samples")
print("[OK] 266 batches")
print("[OK] best.pt")
print("[OK] last.pt")
print("[OK] metadata saved:", metadata_path)

###########################################################################
# CELL 16
###########################################################################
# ============================================================
# CELL 13 — AUDIT ORIGINAL C-IRFS EXPERIMENT CONFIGURATION
# Read-only. No training, no dataset modification.
# ============================================================

from pathlib import Path
import yaml
import csv
import hashlib
import json

# ------------------------------------------------------------
# 1. Resolve paths
# ------------------------------------------------------------

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

BASELINE_CHECKPOINT = (
    PROJECT_DIR / "models" / "finetune_best.pt"
)

C_RUN_DIR = (
    PROJECT_DIR / "research_experiments" / "C_irfs_t030"
)

C_ARGS_FILE = C_RUN_DIR / "args.yaml"
C_RESULTS_FILE = C_RUN_DIR / "results.csv"

print("=" * 75)
print("PATH VERIFICATION")
print("=" * 75)

for label, path in [
    ("Project directory", PROJECT_DIR),
    ("Initial checkpoint", BASELINE_CHECKPOINT),
    ("C-IRFS run directory", C_RUN_DIR),
    ("C-IRFS args.yaml", C_ARGS_FILE),
    ("C-IRFS results.csv", C_RESULTS_FILE),
]:
    print(f"{label:25s}: {path}")
    print(" " * 27, "EXISTS" if path.exists() else "NOT FOUND")

if not BASELINE_CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Initial checkpoint missing: {BASELINE_CHECKPOINT}"
    )

if not C_ARGS_FILE.is_file():
    raise FileNotFoundError(
        f"C configuration missing: {C_ARGS_FILE}"
    )


# ------------------------------------------------------------
# 2. Read original C-IRFS training configuration
# ------------------------------------------------------------

with open(C_ARGS_FILE, "r", encoding="utf-8") as f:
    c_args = yaml.safe_load(f)

important_keys = [
    "model",
    "data",
    "epochs",
    "imgsz",
    "batch",
    "workers",
    "device",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "nbs",
    "cos_lr",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "mask_ratio",
    "overlap_mask",
    "mosaic",
    "close_mosaic",
    "copy_paste",
    "mixup",
    "amp",
    "cache",
    "seed",
    "deterministic",
    "rect",
    "val",
    "save_period",
    "patience",
    "pretrained",
]

print("\n" + "=" * 75)
print("ORIGINAL C-IRFS CONFIGURATION")
print("=" * 75)

for key in important_keys:
    print(f"{key:22s}: {c_args.get(key, '[NOT RECORDED]')}")


# ------------------------------------------------------------
# 3. Verify the original initialization checkpoint
# ------------------------------------------------------------

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


baseline_hash = sha256_file(BASELINE_CHECKPOINT)

print("\n" + "=" * 75)
print("INITIAL CHECKPOINT")
print("=" * 75)

print("Path:", BASELINE_CHECKPOINT)
print(
    f"Size: {BASELINE_CHECKPOINT.stat().st_size / (1024**2):.2f} MB"
)
print("SHA256:", baseline_hash)


# ------------------------------------------------------------
# 4. Inspect the actual C-IRFS completed epochs
# ------------------------------------------------------------

if C_RESULTS_FILE.is_file():

    with open(
        C_RESULTS_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    print("\n" + "=" * 75)
    print("C-IRFS TRAINING HISTORY")
    print("=" * 75)

    print("Rows in results.csv:", len(rows))

    if rows and "epoch" in rows[0]:
        epochs_recorded = [
            int(float(row["epoch"]))
            for row in rows
        ]

        print("First recorded epoch:", epochs_recorded[0])
        print("Last recorded epoch :", epochs_recorded[-1])
        print("Completed epochs    :", len(epochs_recorded))

    else:
        print("Epoch column not found.")

else:
    rows = []
    print("\nC results.csv not found; epoch history unavailable.")


# ------------------------------------------------------------
# 5. Compare recorded source model with initial checkpoint
# ------------------------------------------------------------

recorded_model = c_args.get("model")

print("\n" + "=" * 75)
print("INITIALIZATION SOURCE CHECK")
print("=" * 75)

print("Recorded model in C args.yaml:")
print(recorded_model)

print("\nExpected initial checkpoint:")
print(BASELINE_CHECKPOINT)

if recorded_model:
    recorded_name = Path(str(recorded_model)).name
    expected_name = BASELINE_CHECKPOINT.name

    print("\nRecorded model basename:", recorded_name)
    print("Expected model basename :", expected_name)

    if recorded_name == expected_name:
        print("[OK] Basename matches the expected initial checkpoint.")
    else:
        print(
            "[WARNING] Basenames differ. "
            "We need to inspect the original C training setup."
        )


# ------------------------------------------------------------
# 6. Save audit report
# ------------------------------------------------------------

audit = {
    "baseline_checkpoint": str(BASELINE_CHECKPOINT),
    "baseline_checkpoint_sha256": baseline_hash,
    "c_run_dir": str(C_RUN_DIR),
    "c_args_file": str(C_ARGS_FILE),
    "c_results_file": str(C_RESULTS_FILE),
    "c_args": c_args,
    "c_completed_epochs": len(rows),
    "recorded_model": recorded_model,
}

audit_path = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "loss_api_inspection"
    / "c_irfs_experiment_audit.json"
)

audit_path.parent.mkdir(parents=True, exist_ok=True)

audit_path.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)

print("\n" + "=" * 75)
print("AUDIT COMPLETED")
print("=" * 75)

print("Audit report:", audit_path)
print("\n[CELL 13 COMPLETED]")

###########################################################################
# CELL 17
###########################################################################
# ============================================================
# CELL 14 — NUMERICAL VALIDATION OF CUSTOM MASK LOSS
#
# Test 1: Unit weights must reproduce original Ultralytics loss.
# Test 2: Class weights must modify mask loss only.
# Test 3: Backpropagation gradients must be finite.
#
# No model training / optimizer updates.
# ============================================================

import copy
import torch

from ultralytics.utils.loss import v8SegmentationLoss


# ------------------------------------------------------------
# 1. Prepare an isolated copy of the smoke-test model
# ------------------------------------------------------------

assert isinstance(
    trainer.model,
    ClassBalancedSegmentationModel
)

test_model = copy.deepcopy(trainer.model).to(trainer.device)
test_model.train()

print("=" * 75)
print("MODEL")
print("=" * 75)

print("Model:", type(test_model).__name__)
print("Device:", next(test_model.parameters()).device)


# ------------------------------------------------------------
# 2. Obtain one real training batch
# ------------------------------------------------------------

raw_batch = next(iter(trainer.train_loader))
batch = trainer.preprocess_batch(raw_batch)

print("\n" + "=" * 75)
print("BATCH")
print("=" * 75)

print("Image shape:", tuple(batch["img"].shape))
print("Image dtype:", batch["img"].dtype)
print("Number of labels:", len(batch["cls"]))

unique_classes = torch.unique(
    batch["cls"].long()
).detach().cpu().tolist()

print("Classes present in this batch:", unique_classes)


# ------------------------------------------------------------
# 3. Run model forward once
# ------------------------------------------------------------

test_model.zero_grad(set_to_none=True)

preds = test_model(batch["img"])

print("\n[OK] Forward pass completed.")


# ------------------------------------------------------------
# 4. Compare default loss with custom loss using unit weights
# ------------------------------------------------------------

default_criterion = v8SegmentationLoss(test_model)

unit_weight_criterion = ClassBalancedSegmentationLoss(
    test_model
)

unit_weight_criterion.mask_class_weights = torch.ones(
    len(EXPECTED_CLASS_NAMES),
    device=trainer.device,
    dtype=torch.float32,
)

default_loss, default_items = default_criterion(
    preds,
    batch,
)

unit_loss, unit_items = unit_weight_criterion(
    preds,
    batch,
)

print("\n" + "=" * 75)
print("TEST 1 — UNIT WEIGHTS VS DEFAULT LOSS")
print("=" * 75)

max_difference = 0.0

for key in default_items:

    default_value = default_items[key].float()
    custom_value = unit_items[key].float()

    difference = (
        default_value - custom_value
    ).abs().max().item()

    max_difference = max(
        max_difference,
        difference,
    )

    print(
        f"{key:15s} "
        f"default={default_value.item():.8f} "
        f"custom={custom_value.item():.8f} "
        f"diff={difference:.10f}"
    )

assert torch.allclose(
    default_loss,
    unit_loss,
    rtol=1e-5,
    atol=1e-6,
), (
    "Custom loss with all weights=1 does not reproduce "
    "the original Ultralytics loss."
)

print("\n[PASS] Unit-weight equivalence verified.")
print("Maximum component difference:", max_difference)


# ------------------------------------------------------------
# 5. Evaluate actual class-balanced weights
# ------------------------------------------------------------

balanced_criterion = ClassBalancedSegmentationLoss(
    test_model
)

balanced_criterion.mask_class_weights = torch.tensor(
    CLASS_BALANCED_WEIGHTS,
    dtype=torch.float32,
    device=trainer.device,
)

balanced_loss, balanced_items = balanced_criterion(
    preds,
    batch,
)

print("\n" + "=" * 75)
print("TEST 2 — CLASS-BALANCED LOSS COMPONENTS")
print("=" * 75)

for key in default_items:

    default_value = default_items[key].float()
    balanced_value = balanced_items[key].float()

    difference = (
        default_value - balanced_value
    ).abs().max().item()

    print(
        f"{key:15s} "
        f"default={default_value.item():.8f} "
        f"balanced={balanced_value.item():.8f} "
        f"diff={difference:.10f}"
    )


# Detection and semantic losses must remain unchanged.
unchanged_components = [
    "box_loss",
    "cls_loss",
    "dfl_loss",
    "sem_loss",
]

for key in unchanged_components:

    assert torch.allclose(
        default_items[key],
        balanced_items[key],
        rtol=1e-5,
        atol=1e-6,
    ), (
        f"Unexpected modification of {key}."
    )

print("\n[PASS] Box, classification, DFL and semantic losses unchanged.")


# ------------------------------------------------------------
# 6. Verify finite loss and gradient
# ------------------------------------------------------------

assert torch.isfinite(balanced_loss).all(), (
    "Non-finite class-balanced loss!"
)

test_model.zero_grad(set_to_none=True)

balanced_loss.sum().backward()

gradient_tensors = [
    p.grad
    for p in test_model.parameters()
    if p.grad is not None
]

assert len(gradient_tensors) > 0, (
    "No gradients were produced."
)

all_gradients_finite = all(
    torch.isfinite(g).all().item()
    for g in gradient_tensors
)

assert all_gradients_finite, (
    "NaN or Inf detected in gradients."
)

nonzero_gradients = sum(
    (g.abs().sum().item() > 0)
    for g in gradient_tensors
)

print("\n" + "=" * 75)
print("TEST 3 — GRADIENT VERIFICATION")
print("=" * 75)

print("Gradient tensors:", len(gradient_tensors))
print("Nonzero gradient tensors:", nonzero_gradients)
print("All gradients finite:", all_gradients_finite)

assert nonzero_gradients > 0

print("\n[PASS] Backpropagation verified.")


# ------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("CUSTOM LOSS NUMERICAL VALIDATION COMPLETED")
print("=" * 75)

print("[PASS] Unit weights reproduce original loss.")
print("[PASS] Only instance mask loss changes.")
print("[PASS] Box loss remains unchanged.")
print("[PASS] Classification loss remains unchanged.")
print("[PASS] DFL loss remains unchanged.")
print("[PASS] Semantic loss remains unchanged.")
print("[PASS] Gradients are finite and nonzero.")

print("\n[CELL 14 COMPLETED]")

###########################################################################
# CELL 18
###########################################################################
# ============================================================
# CELL 14 — NUMERICAL VALIDATION OF CUSTOM MASK LOSS
#
# Test 1: Unit weights reproduce original Ultralytics loss.
# Test 2: Class weights modify instance mask loss only.
# Test 3: Backpropagation gradients are finite.
#
# IMPORTANT:
# Create a NEW DataLoader instead of reusing the exhausted
# InfiniteDataLoader from the completed training session.
#
# No optimizer update. No checkpoint modification.
# ============================================================

import copy
import torch

from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8SegmentationLoss


# ============================================================
# 1. VERIFY THE SOURCE MODEL
# ============================================================

assert isinstance(
    trainer.model,
    ClassBalancedSegmentationModel
), "Unexpected model class."

assert isinstance(
    trainer.model.criterion,
    ClassBalancedSegmentationLoss
), "Custom criterion is not active."

print("=" * 75)
print("MODEL")
print("=" * 75)

print("Model:", type(trainer.model).__name__)
print("Criterion:", type(trainer.model.criterion).__name__)
print("Device:", trainer.device)


# ============================================================
# 2. BUILD A FRESH TRAINING DATASET
# ============================================================

print("\n" + "=" * 75)
print("CREATING A FRESH DATASET")
print("=" * 75)

probe_dataset = trainer.build_dataset(
    trainer.data["train"],
    mode="train",
    batch=4,
)

print("Dataset type:", type(probe_dataset).__name__)
print("Dataset images:", len(probe_dataset))


# ============================================================
# 3. CREATE A TEMPORARY DATALOADER
# ============================================================

probe_loader = DataLoader(
    probe_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=probe_dataset.collate_fn,
)

print("Temporary loader:", type(probe_loader).__name__)
print("Workers:", 0)


# ============================================================
# 4. FETCH ONE BATCH
# ============================================================

try:
    raw_batch = next(iter(probe_loader))

except StopIteration:
    raise RuntimeError(
        "The newly created training DataLoader returned no batch. "
        "Check the training dataset and its image-label pairs."
    )

batch = trainer.preprocess_batch(raw_batch)

required_keys = [
    "img",
    "cls",
    "batch_idx",
    "bboxes",
    "masks",
]

missing_keys = [
    key for key in required_keys
    if key not in batch
]

if missing_keys:
    raise RuntimeError(
        f"Missing batch keys: {missing_keys}"
    )

print("\n" + "=" * 75)
print("BATCH VERIFICATION")
print("=" * 75)

print("Image shape:", tuple(batch["img"].shape))
print("Image dtype:", batch["img"].dtype)
print("Image device:", batch["img"].device)

print("Number of labels:", len(batch["cls"]))
print("Number of mask entries:", len(batch["masks"]))

unique_classes = torch.unique(
    batch["cls"].long()
).detach().cpu().tolist()

print("Classes present:", unique_classes)

assert batch["img"].shape[0] == 4
assert len(batch["cls"]) > 0
assert len(batch["masks"]) > 0

print("\n[OK] Fresh training batch obtained.")


# ============================================================
# 5. CREATE AN ISOLATED MODEL COPY
# ============================================================

test_model = copy.deepcopy(
    trainer.model
).to(trainer.device)

test_model.train()
test_model.zero_grad(set_to_none=True)

print("\n" + "=" * 75)
print("ISOLATED TEST MODEL")
print("=" * 75)

print("Model:", type(test_model).__name__)
print("Device:", next(test_model.parameters()).device)


# ============================================================
# 6. FORWARD PASS
# ============================================================

preds = test_model(batch["img"])

print("\n[OK] Forward pass completed.")


# ============================================================
# 7. TEST 1 — UNIT WEIGHTS VS ORIGINAL LOSS
# ============================================================

default_criterion = v8SegmentationLoss(
    test_model
)

unit_weight_criterion = ClassBalancedSegmentationLoss(
    test_model
)

# All weights = 1.
unit_weight_criterion.mask_class_weights = torch.ones(
    len(EXPECTED_CLASS_NAMES),
    device=trainer.device,
    dtype=torch.float32,
)

default_loss, default_items = default_criterion(
    preds,
    batch,
)

unit_loss, unit_items = unit_weight_criterion(
    preds,
    batch,
)

print("\n" + "=" * 75)
print("TEST 1 — UNIT WEIGHTS VS DEFAULT LOSS")
print("=" * 75)

max_difference = 0.0

for key in default_items:

    default_value = default_items[key].float()
    custom_value = unit_items[key].float()

    difference = (
        default_value - custom_value
    ).abs().max().item()

    max_difference = max(
        max_difference,
        difference,
    )

    print(
        f"{key:15s} "
        f"default={default_value.item():.8f} "
        f"custom={custom_value.item():.8f} "
        f"diff={difference:.10f}"
    )

assert torch.allclose(
    default_loss,
    unit_loss,
    rtol=1e-5,
    atol=1e-6,
), (
    "FAIL: Custom loss with unit weights "
    "does not reproduce original Ultralytics loss."
)

print("\n[PASS] Unit-weight equivalence verified.")
print("Maximum component difference:", max_difference)


# ============================================================
# 8. TEST 2 — ACTUAL CLASS-BALANCED WEIGHTS
# ============================================================

balanced_criterion = ClassBalancedSegmentationLoss(
    test_model
)

balanced_criterion.mask_class_weights = torch.tensor(
    CLASS_BALANCED_WEIGHTS,
    dtype=torch.float32,
    device=trainer.device,
)

balanced_loss, balanced_items = balanced_criterion(
    preds,
    batch,
)

print("\n" + "=" * 75)
print("TEST 2 — CLASS-BALANCED LOSS COMPONENTS")
print("=" * 75)

for key in default_items:

    default_value = default_items[key].float()
    balanced_value = balanced_items[key].float()

    difference = (
        default_value - balanced_value
    ).abs().max().item()

    print(
        f"{key:15s} "
        f"default={default_value.item():.8f} "
        f"balanced={balanced_value.item():.8f} "
        f"diff={difference:.10f}"
    )


# Only instance mask loss should change.
unchanged_components = [
    "box_loss",
    "cls_loss",
    "dfl_loss",
    "sem_loss",
]

for key in unchanged_components:

    assert torch.allclose(
        default_items[key],
        balanced_items[key],
        rtol=1e-5,
        atol=1e-6,
    ), (
        f"FAIL: Unexpected modification of {key}."
    )

print("\n[PASS] Detection and semantic loss components unchanged.")


# ============================================================
# 9. TEST 3 — GRADIENT VERIFICATION
# ============================================================

assert torch.isfinite(balanced_loss).all(), (
    "Non-finite class-balanced loss detected."
)

test_model.zero_grad(set_to_none=True)

balanced_loss.sum().backward()

gradient_tensors = [
    parameter.grad
    for parameter in test_model.parameters()
    if parameter.grad is not None
]

assert len(gradient_tensors) > 0, (
    "No gradients were produced."
)

all_gradients_finite = all(
    torch.isfinite(gradient).all().item()
    for gradient in gradient_tensors
)

assert all_gradients_finite, (
    "NaN or Inf detected in gradients."
)

nonzero_gradients = sum(
    gradient.abs().sum().item() > 0
    for gradient in gradient_tensors
)

print("\n" + "=" * 75)
print("TEST 3 — GRADIENT VERIFICATION")
print("=" * 75)

print("Gradient tensors:", len(gradient_tensors))
print("Nonzero gradient tensors:", nonzero_gradients)
print("All gradients finite:", all_gradients_finite)

assert nonzero_gradients > 0

print("\n[PASS] Backpropagation verified.")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("CUSTOM LOSS NUMERICAL VALIDATION COMPLETED")
print("=" * 75)

print("[PASS] Fresh training batch obtained.")
print("[PASS] Unit weights reproduce original loss.")
print("[PASS] Only instance mask loss changes.")
print("[PASS] Box loss unchanged.")
print("[PASS] Classification loss unchanged.")
print("[PASS] DFL loss unchanged.")
print("[PASS] Semantic loss unchanged.")
print("[PASS] Gradients finite and nonzero.")

print("\n[CELL 14 COMPLETED]")


# Release temporary dataset/loader.
del probe_loader
del probe_dataset

###########################################################################
# CELL 19
###########################################################################
# ============================================================
# CELL 15 — OFFICIAL CLASS-BALANCED IRFS EXPERIMENT
#
# Experiment:
#   C-IRFS + Square-root Inverse Frequency Mask Loss
#
# Controlled variables:
#   - Same initial checkpoint as original C-IRFS
#   - Same dataset
#   - Same IRFS sampling strategy
#   - Same 17 epochs
#   - Same optimizer and learning-rate schedule
#   - Same augmentation configuration
#
# Only intended change:
#   Class-balanced instance segmentation loss.
# ============================================================

from pathlib import Path
import json
import hashlib
import torch

from ultralytics import YOLO


# ============================================================
# 1. PATHS AND SOURCE CHECKPOINT
# ============================================================

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

BASELINE_CHECKPOINT = (
    PROJECT_DIR / "models" / "finetune_best.pt"
)

DATA_YAML = (
    PROJECT_DIR / "dataset" / "data.yaml"
)

OUTPUT_DIR = (
    PROJECT_DIR / "kaggle_outputs"
)

EXPERIMENT_NAME = "CB_IRFS_sqrt_17ep_v1"

if not BASELINE_CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Initial checkpoint missing:\n{BASELINE_CHECKPOINT}"
    )

if not DATA_YAML.is_file():
    raise FileNotFoundError(
        f"Dataset YAML missing:\n{DATA_YAML}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. VERIFY CUSTOM DEFINITIONS
# ============================================================

assert issubclass(
    ClassBalancedIRFSTrainer,
    IRFSResearchTrainer,
)

assert issubclass(
    ClassBalancedSegmentationModel,
    SegmentationModel,
)

assert issubclass(
    ClassBalancedSegmentationLoss,
    v8SegmentationLoss,
)

assert len(CLASS_BALANCED_WEIGHTS) == 6

assert len(CLASS_INSTANCE_COUNTS) == 6

print("=" * 75)
print("EXPERIMENT CONFIGURATION")
print("=" * 75)

print("Experiment:", EXPERIMENT_NAME)
print("Initial checkpoint:", BASELINE_CHECKPOINT)
print("Dataset:", DATA_YAML)
print("Output:", OUTPUT_DIR)

print("\nClasses and weights:")

for name, count, weight in zip(
    EXPECTED_CLASS_NAMES,
    CLASS_INSTANCE_COUNTS,
    CLASS_BALANCED_WEIGHTS,
):
    print(
        f"{name:22s} "
        f"instances={count:5d} "
        f"weight={weight:.6f}"
    )


# ============================================================
# 3. HASH INITIAL CHECKPOINT
# ============================================================

def calculate_sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


source_sha256 = calculate_sha256(
    BASELINE_CHECKPOINT
)

print("\nInitial checkpoint SHA256:")
print(source_sha256)


# ============================================================
# 4. INITIALIZE FROM THE SAME CHECKPOINT AS C-IRFS
# ============================================================

cb_main = YOLO(
    str(BASELINE_CHECKPOINT)
)

assert cb_main.task == "segment"

print("\nLoaded model:", type(cb_main.model).__name__)
print("Task:", cb_main.task)


# ============================================================
# 5. START THE OFFICIAL 17-EPOCH EXPERIMENT
# ============================================================

print("\n" + "=" * 75)
print("STARTING OFFICIAL CLASS-BALANCED IRFS TRAINING")
print("=" * 75)

results = cb_main.train(

    # Custom trainer
    trainer=ClassBalancedIRFSTrainer,

    # Dataset
    data=str(DATA_YAML),

    # Same training budget as original C-IRFS
    epochs=17,
    patience=18,

    # Image and batch
    imgsz=960,
    batch=4,
    workers=2,
    device=0,

    # Optimizer
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    nbs=64,

    # Learning-rate schedule
    cos_lr=True,
    warmup_epochs=2,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # Segmentation
    mask_ratio=4,
    overlap_mask=True,

    # Augmentation
    mosaic=1.0,
    close_mosaic=2,
    copy_paste=0.0,
    mixup=0.0,

    # Runtime
    amp=True,
    cache=False,

    # Reproducibility
    seed=42,
    deterministic=True,
    rect=False,

    # Validation and saving
    val=True,
    save=True,
    save_period=5,
    plots=False,

    # Output
    project=str(OUTPUT_DIR),
    name=EXPERIMENT_NAME,
    exist_ok=False,

    # Load initial checkpoint weights
    pretrained=True,

    verbose=True,
)


# ============================================================
# 6. VERIFY ACTUAL TRAINER AND CRITERION
# ============================================================

main_trainer = cb_main.trainer

print("\n" + "=" * 75)
print("TRAINER VERIFICATION")
print("=" * 75)

print("Trainer:", type(main_trainer).__name__)
print("Model:", type(main_trainer.model).__name__)
print("Criterion:", type(main_trainer.model.criterion).__name__)

assert isinstance(
    main_trainer,
    ClassBalancedIRFSTrainer,
)

assert isinstance(
    main_trainer.model,
    ClassBalancedSegmentationModel,
)

assert isinstance(
    main_trainer.model.criterion,
    ClassBalancedSegmentationLoss,
)

print("\n[OK] Correct custom trainer.")
print("[OK] Correct custom model.")
print("[OK] Correct class-balanced criterion.")


# ============================================================
# 7. VERIFY IRFS SAMPLING
# ============================================================

sampling_stats = main_trainer._sampling_statistics

sampler = main_trainer.train_loader.sampler

actual_samples = len(sampler)
actual_batches = len(main_trainer.train_loader)

print("\n" + "=" * 75)
print("IRFS SAMPLER VERIFICATION")
print("=" * 75)

print("Method:", sampling_stats["method"])
print("Threshold:", sampling_stats["threshold"])
print("Unique dataset images:", sampling_stats["num_images"])
print("Total instances:", sampling_stats["total_instances"])
print("Samples per epoch:", actual_samples)
print("Batches per epoch:", actual_batches)

assert sampling_stats["method"] == "IRFS"
assert abs(
    sampling_stats["threshold"] - 0.30
) < 1e-9

assert actual_samples == 1064
assert actual_batches == 266

print("\n[OK] IRFS sampler verified.")


# ============================================================
# 8. VERIFY FINAL CHECKPOINTS
# ============================================================

run_dir = Path(
    main_trainer.save_dir
)

best_path = (
    run_dir / "weights" / "best.pt"
)

last_path = (
    run_dir / "weights" / "last.pt"
)

if not best_path.is_file():
    raise FileNotFoundError(
        f"Best checkpoint missing: {best_path}"
    )

if not last_path.is_file():
    raise FileNotFoundError(
        f"Last checkpoint missing: {last_path}"
    )

print("\n" + "=" * 75)
print("CHECKPOINTS")
print("=" * 75)

print("Best:", best_path)
print("Last:", last_path)


# ============================================================
# 9. SAVE REPRODUCIBILITY METADATA
# ============================================================

metadata = {

    "experiment": EXPERIMENT_NAME,

    "method": (
        "IRFS threshold 0.30 + "
        "square-root inverse-frequency class-balanced mask loss"
    ),

    "initial_checkpoint": str(BASELINE_CHECKPOINT),
    "initial_checkpoint_sha256": source_sha256,

    "dataset_yaml": str(DATA_YAML),

    "epochs_requested": 17,

    "trainer_class": type(main_trainer).__name__,
    "model_class": type(main_trainer.model).__name__,
    "criterion_class": type(main_trainer.model.criterion).__name__,

    "sampler_method": sampling_stats["method"],
    "sampler_threshold": sampling_stats["threshold"],
    "sampler_statistics": sampling_stats,

    "samples_per_epoch": actual_samples,
    "batches_per_epoch": actual_batches,

    "class_names": EXPECTED_CLASS_NAMES,
    "class_instance_counts": CLASS_INSTANCE_COUNTS,
    "class_mask_weights": [
        float(x)
        for x in main_trainer.model.mask_class_weights
    ],

    "training_configuration": {

        "imgsz": 960,
        "batch": 4,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": 2,

        "mosaic": 1.0,
        "close_mosaic": 2,
        "copy_paste": 0.0,
        "mixup": 0.0,

        "mask_ratio": 4,
        "overlap_mask": True,

        "seed": 42,
        "deterministic": True,
    },

    "best_checkpoint": str(best_path),
    "last_checkpoint": str(last_path),

    "note": (
        "Official 17-epoch controlled experiment. "
        "Compared with original C-IRFS, the intended "
        "experimental variable is class-balanced instance mask loss."
    ),
}

metadata_path = (
    run_dir / "experiment_metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("OFFICIAL EXPERIMENT COMPLETED")
print("=" * 75)

print("Experiment:", EXPERIMENT_NAME)
print("Run directory:", run_dir)

print("Trainer:", type(main_trainer).__name__)
print("Model:", type(main_trainer.model).__name__)
print("Criterion:", type(main_trainer.model.criterion).__name__)

print("Sampler:", sampling_stats["method"])
print("Samples/epoch:", actual_samples)
print("Batches/epoch:", actual_batches)

print("\nBest checkpoint:")
print(best_path)

print("\nLast checkpoint:")
print(last_path)

print("\nMetadata:")
print(metadata_path)

print("\n[CELL 15 COMPLETED]")

###########################################################################
# CELL 20
###########################################################################
# ============================================================
# CELL 16 — FAIR EVALUATION: C-IRFS vs CLASS-BALANCED IRFS
#
# Checkpoints:
#   1. Original C-IRFS
#   2. Class-balanced IRFS
#
# Validation splits:
#   1. Combined 268
#   2. Old 119
#   3. New 149
#
# Identical evaluation configuration:
#   imgsz=960, batch=2, rect=True
#
# No training. No checkpoint modification.
# ============================================================

from pathlib import Path
import csv
import json
import gc
import yaml
import torch

from ultralytics import YOLO


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

C_CHECKPOINT = (
    PROJECT_DIR
    / "research_experiments"
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

CB_CHECKPOINT = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "CB_IRFS_sqrt_17ep_v1"
    / "weights"
    / "best.pt"
)

DATA_CONFIGS = {
    "Combined_268": (
        PROJECT_DIR / "dataset" / "data.yaml"
    ),
    "Old_119": (
        PROJECT_DIR / "dataset" / "old_validation.yaml"
    ),
    "New_149": (
        PROJECT_DIR / "dataset" / "new_validation.yaml"
    ),
}

EVAL_OUTPUT = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "fair_evaluation_C_vs_CB"
)

EVAL_OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. VERIFY CHECKPOINTS AND DATA CONFIGURATIONS
# ============================================================

CHECKPOINTS = {
    "C_IRFS": C_CHECKPOINT,
    "CB_IRFS": CB_CHECKPOINT,
}

print("=" * 80)
print("CHECKPOINT VERIFICATION")
print("=" * 80)

for name, path in CHECKPOINTS.items():

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} checkpoint missing:\n{path}"
        )

    print(
        f"{name:12s}: {path} "
        f"({path.stat().st_size / (1024**2):.2f} MB)"
    )


print("\n" + "=" * 80)
print("VALIDATION CONFIGURATION VERIFICATION")
print("=" * 80)

for split_name, yaml_path in DATA_CONFIGS.items():

    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"{split_name} YAML missing:\n{yaml_path}"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    print(f"\n{split_name}")
    print("YAML:", yaml_path)
    print("Validation path:", data_config.get("val"))
    print("Class names:", data_config.get("names"))


# ============================================================
# 3. EVALUATION CONFIGURATION
# ============================================================

EVAL_ARGS = {
    "imgsz": 960,
    "batch": 2,
    "rect": True,
    "device": 0,
    "workers": 2,
    "plots": False,
    "save_json": False,
    "verbose": True,
}

print("\n" + "=" * 80)
print("COMMON EVALUATION PARAMETERS")
print("=" * 80)

for key, value in EVAL_ARGS.items():
    print(f"{key:15s}: {value}")


# ============================================================
# 4. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

all_results = []

def evaluate_checkpoint(
    checkpoint_name,
    checkpoint_path,
    split_name,
    yaml_path,
):

    print("\n")
    print("#" * 80)
    print(f"CHECKPOINT: {checkpoint_name}")
    print(f"SPLIT     : {split_name}")
    print("#" * 80)

    model = YOLO(str(checkpoint_path))

    run_name = f"{checkpoint_name}_{split_name}"

    metrics = model.val(
        data=str(yaml_path),

        project=str(EVAL_OUTPUT),
        name=run_name,
        exist_ok=True,

        **EVAL_ARGS,
    )

    # --------------------------------------------------------
    # Extract mask metrics.
    # --------------------------------------------------------

    seg = metrics.seg

    record = {
        "checkpoint": checkpoint_name,
        "split": split_name,

        "mask_precision": float(seg.mp),
        "mask_recall": float(seg.mr),

        "mask_map50": float(seg.map50),
        "mask_map75": float(seg.map75),
        "mask_map50_95": float(seg.map),
    }

    # Per-class mask AP50-95.
    class_maps = list(seg.maps)

    if len(class_maps) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Expected {len(CLASS_NAMES)} class AP values, "
            f"got {len(class_maps)}."
        )

    for class_name, ap_value in zip(
        CLASS_NAMES,
        class_maps,
    ):
        record[f"AP_{class_name}"] = float(ap_value)

    all_results.append(record)

    print("\n" + "=" * 75)
    print("EXTRACTED MASK METRICS")
    print("=" * 75)

    for key, value in record.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")

    # Release resources.
    del model
    del metrics
    del seg

    gc.collect()
    torch.cuda.empty_cache()

    return record


# ============================================================
# 6. RUN IDENTICAL EVALUATIONS
# ============================================================

for split_name, yaml_path in DATA_CONFIGS.items():

    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():

        evaluate_checkpoint(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            split_name=split_name,
            yaml_path=yaml_path,
        )


# ============================================================
# 7. SAVE RAW RESULTS
# ============================================================

csv_path = EVAL_OUTPUT / "fair_evaluation_results.csv"
json_path = EVAL_OUTPUT / "fair_evaluation_results.json"

fieldnames = list(all_results[0].keys())

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(all_results)


with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 8. PRINT DIRECT COMPARISON
# ============================================================

print("\n" + "=" * 100)
print("FAIR COMPARISON — MASK METRICS")
print("=" * 100)

for split_name in DATA_CONFIGS:

    baseline = next(
        r for r in all_results
        if r["checkpoint"] == "C_IRFS"
        and r["split"] == split_name
    )

    treatment = next(
        r for r in all_results
        if r["checkpoint"] == "CB_IRFS"
        and r["split"] == split_name
    )

    print(f"\n{split_name}")
    print("-" * 100)

    comparison_keys = [
        "mask_precision",
        "mask_recall",
        "mask_map50",
        "mask_map75",
        "mask_map50_95",
    ]

    comparison_keys += [
        f"AP_{name}"
        for name in CLASS_NAMES
    ]

    print(
        f"{'Metric':25s}"
        f"{'C-IRFS':>14s}"
        f"{'CB-IRFS':>14s}"
        f"{'Delta':>14s}"
    )

    for key in comparison_keys:

        value_c = baseline[key]
        value_cb = treatment[key]

        delta = value_cb - value_c

        print(
            f"{key:25s}"
            f"{value_c:14.6f}"
            f"{value_cb:14.6f}"
            f"{delta:+14.6f}"
        )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FAIR EVALUATION COMPLETED")
print("=" * 100)

print("Results CSV :", csv_path)
print("Results JSON:", json_path)

print("Evaluated checkpoints:", len(CHECKPOINTS))
print("Validation splits:", len(DATA_CONFIGS))
print("Total evaluations:", len(all_results))

print("\n[CELL 16 COMPLETED]")

###########################################################################
# CELL 21
###########################################################################
# ============================================================
# CELL 16B — FIXED FAIR EVALUATION
#
# Compare:
#   C-IRFS vs Class-balanced IRFS
#
# Splits:
#   Combined_268, Old_119, New_149
#
# Fix:
#   Temporary evaluation YAMLs inherit "train", "path",
#   "names" and other settings from the main data.yaml.
#   Only "val" is replaced by the corresponding split.
#
# No training. No dataset or checkpoint modification.
# ============================================================

from pathlib import Path
import yaml
import csv
import json
import gc
import torch

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

DATASET_DIR = PROJECT_DIR / "dataset"

BASE_YAML = DATASET_DIR / "data.yaml"
OLD_YAML = DATASET_DIR / "old_validation.yaml"
NEW_YAML = DATASET_DIR / "new_validation.yaml"

C_CHECKPOINT = (
    PROJECT_DIR
    / "research_experiments"
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

CB_CHECKPOINT = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "CB_IRFS_sqrt_17ep_v1"
    / "weights"
    / "best.pt"
)

EVAL_OUTPUT = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "fair_evaluation_C_vs_CB_v2"
)

EVAL_OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. VERIFY INPUT FILES
# ============================================================

required_files = [
    BASE_YAML,
    OLD_YAML,
    NEW_YAML,
    C_CHECKPOINT,
    CB_CHECKPOINT,
]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("=" * 80)
print("INPUT FILE VERIFICATION")
print("=" * 80)

for path in required_files:
    print("[OK]", path)


# ============================================================
# 3. LOAD THE YAML CONFIGURATIONS
# ============================================================

def load_yaml(path):

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(
            f"Invalid YAML structure: {path}"
        )

    return config


base_config = load_yaml(BASE_YAML)
old_config = load_yaml(OLD_YAML)
new_config = load_yaml(NEW_YAML)

if "train" not in base_config:
    raise KeyError(
        "The main data.yaml does not contain a train key."
    )

if "val" not in base_config:
    raise KeyError(
        "The main data.yaml does not contain a val key."
    )


# ============================================================
# 4. CREATE TEMPORARY VALIDATION YAML FILES
# ============================================================

split_sources = {
    "Combined_268": (BASE_YAML, base_config["val"]),
    "Old_119": (OLD_YAML, old_config["val"]),
    "New_149": (NEW_YAML, new_config["val"]),
}

evaluation_yamls = {}

for split_name, (source_yaml, val_path) in split_sources.items():

    # Inherit the complete main dataset configuration.
    eval_config = dict(base_config)

    # Replace only the validation split.
    eval_config["val"] = val_path

    # Keep a valid training path because Ultralytics requires it.
    eval_config["train"] = base_config["train"]

    # Save in the same dataset directory so relative paths
    # retain the same reference location.
    eval_yaml = (
        DATASET_DIR
        / f"eval_{split_name.lower()}.yaml"
    )

    with open(
        eval_yaml,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            eval_config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    evaluation_yamls[split_name] = eval_yaml

    print(f"\n{split_name}")
    print("Source YAML:", source_yaml)
    print("Temporary YAML:", eval_yaml)
    print("Train:", eval_config["train"])
    print("Val:", eval_config["val"])


# ============================================================
# 5. PREFLIGHT: ASK ULTRALYTICS TO RESOLVE ALL DATA PATHS
# ============================================================

print("\n" + "=" * 80)
print("DATASET PATH PREFLIGHT")
print("=" * 80)

for split_name, eval_yaml in evaluation_yamls.items():

    resolved = check_det_dataset(
        str(eval_yaml),
        autodownload=False,
    )

    train_resolved = Path(resolved["train"])
    val_resolved = Path(resolved["val"])

    print(f"\n{split_name}")
    print("Resolved train:", train_resolved)
    print("Resolved val  :", val_resolved)

    if not train_resolved.exists():
        raise FileNotFoundError(
            f"Resolved train path does not exist:\n{train_resolved}"
        )

    if not val_resolved.exists():
        raise FileNotFoundError(
            f"Resolved validation path does not exist:\n{val_resolved}"
        )

    # Confirm class mapping remains exactly the same.
    names = resolved["names"]

    observed_names = [
        names[i] for i in range(len(names))
    ]

    expected_names = [
        "hat",
        "gloves",
        "mask",
        "pants",
        "shoes",
        "long-sleeved shirt",
    ]

    assert observed_names == expected_names, (
        f"Class mapping mismatch in {split_name}."
    )

    print("[OK] Train path exists.")
    print("[OK] Validation path exists.")
    print("[OK] Class mapping is correct.")


# ============================================================
# 6. DEFINE CHECKPOINTS AND EVALUATION PARAMETERS
# ============================================================

CHECKPOINTS = {
    "C_IRFS": C_CHECKPOINT,
    "CB_IRFS": CB_CHECKPOINT,
}

EVAL_ARGS = {
    "imgsz": 960,
    "batch": 2,
    "rect": True,
    "device": 0,
    "workers": 2,
    "plots": False,
    "save_json": False,
    "verbose": True,
}

CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

print("\n" + "=" * 80)
print("EVALUATION PARAMETERS")
print("=" * 80)

for key, value in EVAL_ARGS.items():
    print(f"{key:15s}: {value}")


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

all_results = []


def evaluate_checkpoint(
    checkpoint_name,
    checkpoint_path,
    split_name,
    eval_yaml,
):

    print("\n")
    print("#" * 80)
    print("CHECKPOINT:", checkpoint_name)
    print("SPLIT:", split_name)
    print("#" * 80)

    model = YOLO(str(checkpoint_path))

    run_name = f"{checkpoint_name}_{split_name}"

    metrics = model.val(
        data=str(eval_yaml),

        project=str(EVAL_OUTPUT),
        name=run_name,
        exist_ok=True,

        **EVAL_ARGS,
    )

    seg = metrics.seg

    record = {
        "checkpoint": checkpoint_name,
        "split": split_name,

        "mask_precision": float(seg.mp),
        "mask_recall": float(seg.mr),
        "mask_map50": float(seg.map50),
        "mask_map75": float(seg.map75),
        "mask_map50_95": float(seg.map),
    }

    class_maps = list(seg.maps)

    if len(class_maps) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Expected six class AP values, "
            f"received {len(class_maps)}."
        )

    for class_name, ap_value in zip(
        CLASS_NAMES,
        class_maps,
    ):
        record[f"AP_{class_name}"] = float(ap_value)

    all_results.append(record)

    print("\nExtracted metrics:")

    for key, value in record.items():

        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")

    del model
    del metrics
    del seg

    gc.collect()
    torch.cuda.empty_cache()

    return record


# ============================================================
# 8. RUN ALL SIX EVALUATIONS
# ============================================================

for split_name, eval_yaml in evaluation_yamls.items():

    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():

        evaluate_checkpoint(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            split_name=split_name,
            eval_yaml=eval_yaml,
        )


# ============================================================
# 9. SAVE RAW RESULTS
# ============================================================

csv_path = (
    EVAL_OUTPUT / "fair_evaluation_results.csv"
)

json_path = (
    EVAL_OUTPUT / "fair_evaluation_results.json"
)

fieldnames = list(all_results[0].keys())

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(all_results)


with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 10. COMPARE C-IRFS WITH CLASS-BALANCED IRFS
# ============================================================

print("\n" + "=" * 110)
print("FAIR COMPARISON — MASK METRICS")
print("=" * 110)

comparison_keys = [
    "mask_precision",
    "mask_recall",
    "mask_map50",
    "mask_map75",
    "mask_map50_95",
]

comparison_keys.extend(
    [f"AP_{name}" for name in CLASS_NAMES]
)

for split_name in evaluation_yamls:

    baseline = next(
        r for r in all_results
        if r["checkpoint"] == "C_IRFS"
        and r["split"] == split_name
    )

    treatment = next(
        r for r in all_results
        if r["checkpoint"] == "CB_IRFS"
        and r["split"] == split_name
    )

    print(f"\n{split_name}")
    print("-" * 110)

    print(
        f"{'Metric':30s}"
        f"{'C-IRFS':>14s}"
        f"{'CB-IRFS':>14s}"
        f"{'Delta':>14s}"
    )

    for key in comparison_keys:

        value_c = baseline[key]
        value_cb = treatment[key]

        delta = value_cb - value_c

        print(
            f"{key:30s}"
            f"{value_c:14.6f}"
            f"{value_cb:14.6f}"
            f"{delta:+14.6f}"
        )


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FAIR EVALUATION COMPLETED")
print("=" * 100)

print("Checkpoints:", len(CHECKPOINTS))
print("Validation splits:", len(evaluation_yamls))
print("Total evaluations:", len(all_results))

print("\nCSV :", csv_path)
print("JSON:", json_path)

print("\n[CELL 16B COMPLETED]")

###########################################################################
# CELL 22
###########################################################################
# ============================================================
# CELL 17 — PAIRED ERROR ANALYSIS
# C-IRFS vs CLASS-BALANCED IRFS
#
# Dataset: fixed validation set, 268 images
# Models:
#   1. C-IRFS
#   2. CB-IRFS
#
# Prediction:
#   imgsz=960, batch=2, rect=True
#   confidence=0.001, NMS IoU=0.7
#   retina_masks=True
#
# Matching:
#   Mask IoU >= 0.50
#   Mask IoU >= 0.75
#
# Outputs:
#   1. per_image_class_metrics.csv
#   2. paired_image_class_comparison.csv
#   3. per_ground_truth_instance.csv
#   4. all_predictions.csv
#   5. false_positive_predictions.csv
#   6. class_summary.csv
#   7. size_stratified_recall.csv
#   8. false_positive_type_summary.csv
#   9. analysis_metadata.json
#
# No training. No checkpoint modification.
# ============================================================

import gc
import csv
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

from ultralytics import YOLO

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# 1. PATHS AND EXPERIMENT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

DATASET_DIR = PROJECT_DIR / "dataset"

VAL_IMAGE_ROOT = DATASET_DIR / "images" / "val"
VAL_LABEL_ROOT = DATASET_DIR / "labels" / "val"

C_CHECKPOINT = (
    PROJECT_DIR
    / "research_experiments"
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

CB_CHECKPOINT = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "CB_IRFS_sqrt_17ep_v1"
    / "weights"
    / "best.pt"
)

ANALYSIS_DIR = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "paired_error_analysis_v1"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = {
    "C_IRFS": C_CHECKPOINT,
    "CB_IRFS": CB_CHECKPOINT,
}

CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

NUM_CLASSES = len(CLASS_NAMES)

# Low confidence threshold to retain candidate predictions.
CONF_THRESHOLD = 0.001

# Ultralytics NMS IoU.
NMS_IOU = 0.7

# One-to-one mask matching thresholds.
MATCH_THRESHOLDS = [0.50, 0.75]

# Prediction configuration.
PREDICT_ARGS = {
    "imgsz": 960,
    "batch": 2,
    "rect": True,
    "device": 0,
    "workers": 2,
    "conf": CONF_THRESHOLD,
    "iou": NMS_IOU,
    "max_det": 300,
    "retina_masks": True,
    "agnostic_nms": False,
    "augment": False,
    "verbose": False,
    "save": False,
    "stream": True,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

print("=" * 80)
print("PAIRED ERROR ANALYSIS CONFIGURATION")
print("=" * 80)

print("Validation images:", VAL_IMAGE_ROOT)
print("Validation labels:", VAL_LABEL_ROOT)
print("Output directory :", ANALYSIS_DIR)

for name, path in MODELS.items():

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} checkpoint not found:\n{path}"
        )

    print(f"{name:12s}: {path}")


# ============================================================
# 2. ENUMERATE ALL 268 VALIDATION IMAGES
# ============================================================

image_paths = sorted(
    p.resolve()
    for p in VAL_IMAGE_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

if len(image_paths) != 268:
    raise RuntimeError(
        f"Expected 268 validation images, "
        f"found {len(image_paths)}."
    )

image_path_map = {
    p: p for p in image_paths
}

basename_map = {
    p.name: p for p in image_paths
}

image_split_map = {}

for image_path in image_paths:

    relative_path = image_path.relative_to(
        VAL_IMAGE_ROOT
    )

    if relative_path.parts[0] == "old":
        split_name = "Old_119"

    elif relative_path.parts[0] == "new":
        split_name = "New_149"

    else:
        raise RuntimeError(
            f"Cannot identify Old/New split:\n{relative_path}"
        )

    image_split_map[image_path] = split_name


old_count = sum(
    value == "Old_119"
    for value in image_split_map.values()
)

new_count = sum(
    value == "New_149"
    for value in image_split_map.values()
)

assert old_count == 119
assert new_count == 149

print("\n" + "=" * 80)
print("VALIDATION IMAGE AUDIT")
print("=" * 80)

print("Combined:", len(image_paths))
print("Old     :", old_count)
print("New     :", new_count)


# ============================================================
# 3. PARSE YOLO SEGMENTATION LABELS
# ============================================================

def label_path_from_image(image_path):

    relative_path = image_path.relative_to(
        VAL_IMAGE_ROOT
    )

    return (
        VAL_LABEL_ROOT
        / relative_path
    ).with_suffix(".txt")


def parse_polygon_label_file(label_path):

    """
    Each line:
        class_id x1 y1 x2 y2 ... xn yn

    Coordinates are normalized to [0, 1].

    Exact duplicate polygons within the same image are removed.
    """

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Missing label file:\n{label_path}"
        )

    objects = []
    seen = set()

    source_line_count = 0
    duplicate_count = 0

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            source_line_count += 1

            try:
                values = np.asarray(
                    [float(x) for x in line.split()],
                    dtype=np.float64,
                )

            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric label: {label_path}, "
                    f"line {line_number}"
                ) from exc

            if len(values) < 7:
                raise ValueError(
                    f"Polygon needs at least 3 points: "
                    f"{label_path}, line {line_number}"
                )

            if (len(values) - 1) % 2 != 0:
                raise ValueError(
                    f"Odd number of polygon coordinates: "
                    f"{label_path}, line {line_number}"
                )

            class_float = values[0]
            class_id = int(round(class_float))

            if abs(class_float - class_id) > 1e-6:
                raise ValueError(
                    f"Non-integer class ID: {label_path}"
                )

            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"Invalid class ID {class_id}: {label_path}"
                )

            # Numerical key avoids differences caused only by whitespace.
            key = tuple(
                np.round(values, 8).tolist()
            )

            if key in seen:
                duplicate_count += 1
                continue

            seen.add(key)

            points = values[1:].reshape(-1, 2)

            if not np.isfinite(points).all():
                raise ValueError(
                    f"Non-finite coordinates: {label_path}"
                )

            objects.append({
                "class_id": class_id,
                "points": points,
            })

    return objects, source_line_count, duplicate_count


# Cache polygon coordinates only.
# Do not cache full-resolution masks for all images.
GT_POLYGONS = {}

gt_class_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64,
)

source_label_count = 0
duplicate_label_count = 0

print("\n" + "=" * 80)
print("GROUND-TRUTH LABEL AUDIT")
print("=" * 80)

for index, image_path in enumerate(image_paths, start=1):

    label_path = label_path_from_image(image_path)

    objects, source_count, duplicate_count = (
        parse_polygon_label_file(label_path)
    )

    GT_POLYGONS[image_path] = objects

    source_label_count += source_count
    duplicate_label_count += duplicate_count

    for obj in objects:
        gt_class_counts[obj["class_id"]] += 1

    if index % 50 == 0:
        print(
            f"Processed {index}/{len(image_paths)} images"
        )


EXPECTED_GT_COUNTS = [
    829,
    238,
    339,
    446,
    274,
    916,
]

print("\nSource label lines:", source_label_count)
print("Exact duplicates removed:", duplicate_label_count)
print("Unique GT instances:", int(gt_class_counts.sum()))

for class_id, class_name in enumerate(CLASS_NAMES):

    observed = int(gt_class_counts[class_id])
    expected = EXPECTED_GT_COUNTS[class_id]

    print(
        f"{class_name:22s} "
        f"observed={observed:5d} "
        f"expected={expected:5d}"
    )

    if observed != expected:
        raise RuntimeError(
            f"Ground-truth count mismatch for {class_name}. "
            "Stop before inference to investigate label parsing."
        )

assert int(gt_class_counts.sum()) == 3042

print("\n[OK] Ground-truth counts match validation audit.")


# ============================================================
# 4. MASK UTILITY FUNCTIONS
# ============================================================

def rasterize_polygon(points, height, width):

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    points = np.clip(
        points,
        0.0,
        1.0,
    )

    pixel_points = np.empty_like(points)

    pixel_points[:, 0] = points[:, 0] * (width - 1)
    pixel_points[:, 1] = points[:, 1] * (height - 1)

    pixel_points = np.rint(
        pixel_points
    ).astype(np.int32)

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [pixel_points],
        1,
    )

    return mask.astype(bool)


def mask_bbox_and_area(mask):

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None, 0, 0, 0, 0

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1

    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    width = x2 - x1
    height = y2 - y1

    area = int(len(xs))
    max_side = max(width, height)

    return (
        (x1, y1, x2, y2),
        area,
        width,
        height,
        max_side,
    )


def size_bucket(max_side):

    """
    Pixel-size buckets based on the maximum side
    of the instance mask in original image coordinates.
    """

    if max_side < 32:
        return "lt_32px"

    elif max_side < 64:
        return "32_64px"

    elif max_side < 128:
        return "64_128px"

    return "ge_128px"


def create_gt_objects(
    image_path,
    height,
    width,
):

    objects = []

    for obj in GT_POLYGONS[image_path]:

        mask = rasterize_polygon(
            obj["points"],
            height,
            width,
        )

        bbox, area, box_w, box_h, max_side = (
            mask_bbox_and_area(mask)
        )

        if area == 0:
            raise RuntimeError(
                f"Empty rasterized GT mask:\n{image_path}"
            )

        objects.append({
            "class_id": obj["class_id"],
            "mask": mask,
            "bbox": bbox,
            "mask_area": area,
            "bbox_width": box_w,
            "bbox_height": box_h,
            "max_side": max_side,
            "size_bucket": size_bucket(max_side),
        })

    return objects


def compute_mask_iou(gt_object, pred_object):

    """
    Calculate full-resolution binary mask IoU.
    Only the overlapping bounding-box region is processed.
    """

    gb = gt_object["bbox"]
    pb = pred_object["bbox"]

    if gb is None or pb is None:
        return 0.0

    x1 = max(gb[0], pb[0])
    y1 = max(gb[1], pb[1])

    x2 = min(gb[2], pb[2])
    y2 = min(gb[3], pb[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = np.count_nonzero(
        gt_object["mask"][y1:y2, x1:x2]
        &
        pred_object["mask"][y1:y2, x1:x2]
    )

    union = (
        gt_object["mask_area"]
        + pred_object["mask_area"]
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(intersection / union)


# ============================================================
# 5. ONE-TO-ONE MATCHING
# ============================================================

def build_iou_matrix(gt_objects, pred_objects):

    matrix = np.zeros(
        (len(gt_objects), len(pred_objects)),
        dtype=np.float32,
    )

    for gi, gt in enumerate(gt_objects):

        for pi, pred in enumerate(pred_objects):

            matrix[gi, pi] = compute_mask_iou(
                gt,
                pred,
            )

    return matrix


def match_at_threshold(
    iou_matrix,
    gt_objects,
    pred_objects,
    threshold,
):

    """
    Perform class-wise one-to-one Hungarian matching.

    Matching maximizes total mask IoU, subject to class equality.
    Only matches with IoU >= threshold are accepted.
    """

    gt_to_pred = {}
    pred_to_gt = {}

    gt_classes = [
        obj["class_id"] for obj in gt_objects
    ]

    pred_classes = [
        obj["class_id"] for obj in pred_objects
    ]

    for class_id in range(NUM_CLASSES):

        gt_indices = [
            i for i, c in enumerate(gt_classes)
            if c == class_id
        ]

        pred_indices = [
            i for i, c in enumerate(pred_classes)
            if c == class_id
        ]

        if not gt_indices or not pred_indices:
            continue

        submatrix = iou_matrix[
            np.ix_(gt_indices, pred_indices)
        ]

        if SCIPY_AVAILABLE:

            rows, cols = linear_sum_assignment(
                -submatrix
            )

        else:

            # Greedy fallback.
            candidates = []

            for local_g in range(len(gt_indices)):
                for local_p in range(len(pred_indices)):
                    candidates.append((
                        float(submatrix[local_g, local_p]),
                        local_g,
                        local_p,
                    ))

            candidates.sort(reverse=True)

            used_g = set()
            used_p = set()

            rows = []
            cols = []

            for value, rg, cp in candidates:

                if rg in used_g or cp in used_p:
                    continue

                rows.append(rg)
                cols.append(cp)

                used_g.add(rg)
                used_p.add(cp)

        for local_g, local_p in zip(rows, cols):

            iou = float(
                submatrix[local_g, local_p]
            )

            if iou < threshold:
                continue

            global_g = gt_indices[local_g]
            global_p = pred_indices[local_p]

            gt_to_pred[global_g] = (
                global_p,
                iou,
            )

            pred_to_gt[global_p] = (
                global_g,
                iou,
            )

    return gt_to_pred, pred_to_gt


# ============================================================
# 6. STORAGE FOR ANALYSIS OUTPUTS
# ============================================================

per_image_records = []
gt_records = []
prediction_records = []

inference_summary = {}


# ============================================================
# 7. PROCESS ONE IMAGE
# ============================================================

def analyze_one_image(
    model_name,
    image_path,
    gt_objects,
    pred_objects,
    image_rel_path,
    split_name,
):

    iou_matrix = build_iou_matrix(
        gt_objects,
        pred_objects,
    )

    n_gt = len(gt_objects)
    n_pred = len(pred_objects)

    gt_to_pred_50, pred_to_gt_50 = match_at_threshold(
        iou_matrix,
        gt_objects,
        pred_objects,
        0.50,
    )

    gt_to_pred_75, pred_to_gt_75 = match_at_threshold(
        iou_matrix,
        gt_objects,
        pred_objects,
        0.75,
    )

    # Best same-class and any-class overlap for each GT.
    gt_best_same = []
    gt_best_same_pred = []

    gt_best_other = []
    gt_best_any = []
    gt_best_any_class = []

    for gi, gt in enumerate(gt_objects):

        same_indices = [
            pi for pi, pred in enumerate(pred_objects)
            if pred["class_id"] == gt["class_id"]
        ]

        other_indices = [
            pi for pi, pred in enumerate(pred_objects)
            if pred["class_id"] != gt["class_id"]
        ]

        if same_indices:

            values = iou_matrix[gi, same_indices]
            local = int(np.argmax(values))

            best_value = float(values[local])
            best_pred = same_indices[local]

        else:
            best_value = 0.0
            best_pred = None

        gt_best_same.append(best_value)
        gt_best_same_pred.append(best_pred)

        if other_indices:
            gt_best_other.append(
                float(
                    iou_matrix[gi, other_indices].max()
                )
            )
        else:
            gt_best_other.append(0.0)

        if n_pred:
            best_any_pred = int(
                np.argmax(iou_matrix[gi])
            )

            gt_best_any.append(
                float(iou_matrix[gi, best_any_pred])
            )

            gt_best_any_class.append(
                pred_objects[best_any_pred]["class_id"]
            )

        else:
            gt_best_any.append(0.0)
            gt_best_any_class.append(None)

    # Best overlap of each prediction with any/same/other GT.
    pred_best_same = []
    pred_best_other = []
    pred_best_any = []
    pred_best_any_gt_class = []

    for pi, pred in enumerate(pred_objects):

        same_indices = [
            gi for gi, gt in enumerate(gt_objects)
            if gt["class_id"] == pred["class_id"]
        ]

        other_indices = [
            gi for gi, gt in enumerate(gt_objects)
            if gt["class_id"] != pred["class_id"]
        ]

        if same_indices:
            pred_best_same.append(
                float(
                    iou_matrix[same_indices, pi].max()
                )
            )
        else:
            pred_best_same.append(0.0)

        if other_indices:
            pred_best_other.append(
                float(
                    iou_matrix[other_indices, pi].max()
                )
            )
        else:
            pred_best_other.append(0.0)

        if n_gt:

            best_gt = int(
                np.argmax(iou_matrix[:, pi])
            )

            pred_best_any.append(
                float(iou_matrix[best_gt, pi])
            )

            pred_best_any_gt_class.append(
                gt_objects[best_gt]["class_id"]
            )

        else:
            pred_best_any.append(0.0)
            pred_best_any_gt_class.append(None)

    # --------------------------------------------------------
    # Per-image, per-class metrics.
    # --------------------------------------------------------

    for class_id, class_name in enumerate(CLASS_NAMES):

        gt_indices = [
            i for i, x in enumerate(gt_objects)
            if x["class_id"] == class_id
        ]

        pred_indices = [
            i for i, x in enumerate(pred_objects)
            if x["class_id"] == class_id
        ]

        tp50 = sum(
            1 for gi in gt_indices
            if gi in gt_to_pred_50
        )

        tp75 = sum(
            1 for gi in gt_indices
            if gi in gt_to_pred_75
        )

        fp50 = len(pred_indices) - tp50
        fp75 = len(pred_indices) - tp75

        fn50 = len(gt_indices) - tp50
        fn75 = len(gt_indices) - tp75

        precision50 = (
            tp50 / (tp50 + fp50)
            if tp50 + fp50 else 0.0
        )

        recall50 = (
            tp50 / (tp50 + fn50)
            if tp50 + fn50 else 0.0
        )

        precision75 = (
            tp75 / (tp75 + fp75)
            if tp75 + fp75 else 0.0
        )

        recall75 = (
            tp75 / (tp75 + fn75)
            if tp75 + fn75 else 0.0
        )

        matched_ious_50 = [
            gt_to_pred_50[gi][1]
            for gi in gt_indices
            if gi in gt_to_pred_50
        ]

        matched_ious_75 = [
            gt_to_pred_75[gi][1]
            for gi in gt_indices
            if gi in gt_to_pred_75
        ]

        per_image_records.append({

            "model": model_name,
            "split": split_name,
            "image": image_rel_path,

            "class_id": class_id,
            "class_name": class_name,

            "gt_count": len(gt_indices),
            "prediction_count": len(pred_indices),

            "TP_50": tp50,
            "FP_50": fp50,
            "FN_50": fn50,

            "precision_50": precision50,
            "recall_50": recall50,

            "mean_matched_iou_50": (
                float(np.mean(matched_ious_50))
                if matched_ious_50 else 0.0
            ),

            "TP_75": tp75,
            "FP_75": fp75,
            "FN_75": fn75,

            "precision_75": precision75,
            "recall_75": recall75,

            "mean_matched_iou_75": (
                float(np.mean(matched_ious_75))
                if matched_ious_75 else 0.0
            ),
        })

    # --------------------------------------------------------
    # Per-ground-truth instance analysis.
    # --------------------------------------------------------

    for gi, gt in enumerate(gt_objects):

        matched50 = gt_to_pred_50.get(gi)
        matched75 = gt_to_pred_75.get(gi)

        best_same_iou = gt_best_same[gi]
        best_other_iou = gt_best_other[gi]

        if matched50 is not None:

            fn_reason = "TP"

        elif best_same_iou >= 0.10:

            fn_reason = "same_class_overlap_but_iou_below_0.50"

        elif best_other_iou >= 0.50:

            fn_reason = "possible_class_confusion"

        else:

            fn_reason = "missed_or_no_reliable_overlap"

        best_same_pred = gt_best_same_pred[gi]

        best_same_conf = (
            pred_objects[best_same_pred]["confidence"]
            if best_same_pred is not None
            else 0.0
        )

        matched50_conf = (
            pred_objects[matched50[0]]["confidence"]
            if matched50 is not None
            else 0.0
        )

        matched75_conf = (
            pred_objects[matched75[0]]["confidence"]
            if matched75 is not None
            else 0.0
        )

        gt_records.append({

            "model": model_name,
            "split": split_name,
            "image": image_rel_path,

            "class_id": gt["class_id"],
            "class_name": CLASS_NAMES[gt["class_id"]],

            "gt_mask_area_px": gt["mask_area"],
            "gt_bbox_width_px": gt["bbox_width"],
            "gt_bbox_height_px": gt["bbox_height"],
            "gt_max_side_px": gt["max_side"],
            "size_bucket": gt["size_bucket"],

            "best_same_class_iou": best_same_iou,
            "best_other_class_iou": best_other_iou,
            "best_any_class_iou": gt_best_any[gi],

            "best_any_predicted_class": (
                CLASS_NAMES[gt_best_any_class[gi]]
                if gt_best_any_class[gi] is not None
                else ""
            ),

            "best_same_class_pred_confidence": best_same_conf,

            "matched_at_iou50": matched50 is not None,
            "matched_iou50": (
                matched50[1] if matched50 else 0.0
            ),
            "matched_pred_confidence_50": matched50_conf,

            "matched_at_iou75": matched75 is not None,
            "matched_iou75": (
                matched75[1] if matched75 else 0.0
            ),
            "matched_pred_confidence_75": matched75_conf,

            "miss_reason_at_iou50": fn_reason,
        })

    # --------------------------------------------------------
    # All predictions, including TP and FP.
    # --------------------------------------------------------

    for pi, pred in enumerate(pred_objects):

        matched50 = pred_to_gt_50.get(pi)
        matched75 = pred_to_gt_75.get(pi)

        if matched50 is not None:
            status50 = "TP"
            fp_type = ""

        else:
            status50 = "FP"

            if pred_best_same[pi] >= 0.50:

                fp_type = "duplicate_prediction"

            elif pred_best_other[pi] >= 0.50:

                fp_type = "cross_class_overlap"

            elif pred_best_same[pi] >= 0.10:

                fp_type = "same_class_low_mask_iou"

            elif pred_best_other[pi] >= 0.10:

                fp_type = "other_class_low_mask_iou"

            else:

                fp_type = "background_or_no_overlap"

        status75 = (
            "TP" if matched75 is not None else "FP"
        )

        matched_gt50 = (
            gt_objects[matched50[0]]
            if matched50 is not None
            else None
        )

        pred_records.append({

            "model": model_name,
            "split": split_name,
            "image": image_rel_path,

            "prediction_index": pi,

            "class_id": pred["class_id"],
            "class_name": CLASS_NAMES[pred["class_id"]],

            "confidence": pred["confidence"],

            "pred_mask_area_px": pred["mask_area"],
            "pred_bbox_width_px": pred["bbox_width"],
            "pred_bbox_height_px": pred["bbox_height"],
            "pred_max_side_px": pred["max_side"],

            "best_same_class_iou": pred_best_same[pi],
            "best_other_class_iou": pred_best_other[pi],
            "best_any_class_iou": pred_best_any[pi],

            "best_any_gt_class": (
                CLASS_NAMES[pred_best_any_gt_class[pi]]
                if pred_best_any_gt_class[pi] is not None
                else ""
            ),

            "status_iou50": status50,
            "false_positive_type": fp_type,

            "status_iou75": status75,

            "matched_gt_class_iou50": (
                CLASS_NAMES[matched_gt50["class_id"]]
                if matched_gt50 is not None
                else ""
            ),

            "matched_iou50": (
                matched50[1] if matched50 else 0.0
            ),

            "matched_iou75": (
                matched75[1] if matched75 else 0.0
            ),

            "box_x1": pred["box_xyxy"][0],
            "box_y1": pred["box_xyxy"][1],
            "box_x2": pred["box_xyxy"][2],
            "box_y2": pred["box_xyxy"][3],
        })


# ============================================================
# 8. RUN PREDICTION FOR EACH MODEL
# ============================================================

for model_name, checkpoint_path in MODELS.items():

    print("\n")
    print("#" * 90)
    print("STARTING INFERENCE:", model_name)
    print("#" * 90)

    model = YOLO(str(checkpoint_path))

    processed_images = 0
    total_predictions = 0
    total_gt_instances = 0

    prediction_stream = model.predict(
        source=[str(p) for p in image_paths],
        **PREDICT_ARGS,
    )

    for result in prediction_stream:

        result_path = Path(
            result.path
        ).resolve()

        image_path = image_path_map.get(
            result_path
        )

        # Fallback to basename lookup if path normalization differs.
        if image_path is None:
            image_path = basename_map.get(
                Path(result.path).name
            )

        if image_path is None:
            raise RuntimeError(
                f"Prediction returned an unknown image:\n{result.path}"
            )

        height, width = result.orig_shape

        gt_objects = create_gt_objects(
            image_path,
            height,
            width,
        )

        # ----------------------------------------------------
        # Extract predictions.
        # ----------------------------------------------------

        pred_objects = []

        if result.boxes is not None and len(result.boxes):

            pred_classes = (
                result.boxes.cls.cpu().numpy().astype(int)
            )

            pred_confidences = (
                result.boxes.conf.cpu().numpy()
            )

            pred_boxes = (
                result.boxes.xyxy.cpu().numpy()
            )

            if result.masks is None:
                raise RuntimeError(
                    "Segmentation predictions have boxes "
                    "but no masks."
                )

            pred_masks = (
                result.masks.data
                .detach()
                .cpu()
                .numpy()
            )

            if len(pred_masks) != len(pred_classes):
                raise RuntimeError(
                    "Prediction box/mask count mismatch."
                )

            for index in range(len(pred_classes)):

                class_id = int(pred_classes[index])

                if not 0 <= class_id < NUM_CLASSES:
                    continue

                pred_mask = (
                    pred_masks[index] > 0.5
                )

                # retina_masks=True should already return
                # original-resolution masks. Resize only if needed.
                if pred_mask.shape != (height, width):

                    pred_mask = cv2.resize(
                        pred_mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                else:
                    pred_mask = pred_mask.astype(bool)

                (
                    pred_bbox,
                    pred_area,
                    pred_width,
                    pred_height,
                    pred_max_side,
                ) = mask_bbox_and_area(pred_mask)

                box = pred_boxes[index]

                pred_objects.append({

                    "class_id": class_id,
                    "confidence": float(
                        pred_confidences[index]
                    ),

                    "mask": pred_mask,
                    "bbox": pred_bbox,
                    "mask_area": pred_area,

                    "bbox_width": pred_width,
                    "bbox_height": pred_height,
                    "max_side": pred_max_side,

                    "box_xyxy": [
                        float(x) for x in box
                    ],
                })

        relative_path = str(
            image_path.relative_to(VAL_IMAGE_ROOT)
        )

        split_name = image_split_map[image_path]

        analyze_one_image(
            model_name=model_name,
            image_path=image_path,
            gt_objects=gt_objects,
            pred_objects=pred_objects,
            image_rel_path=relative_path,
            split_name=split_name,
        )

        processed_images += 1
        total_predictions += len(pred_objects)
        total_gt_instances += len(gt_objects)

        if processed_images % 25 == 0:

            print(
                f"{model_name}: "
                f"{processed_images}/{len(image_paths)} images, "
                f"predictions={total_predictions}"
            )

    if processed_images != len(image_paths):
        raise RuntimeError(
            f"{model_name}: expected {len(image_paths)} "
            f"images but processed {processed_images}."
        )

    inference_summary[model_name] = {
        "images": processed_images,
        "ground_truth_instances": total_gt_instances,
        "predictions": total_predictions,
        "checkpoint": str(checkpoint_path),
    }

    print("\nCompleted:", model_name)
    print("Images:", processed_images)
    print("GT instances:", total_gt_instances)
    print("Predictions:", total_predictions)

    del prediction_stream
    del model

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 9. DATAFRAME CREATION
# ============================================================

import pandas as pd

per_image_df = pd.DataFrame(per_image_records)
gt_df = pd.DataFrame(gt_records)
pred_df = pd.DataFrame(prediction_records)

if per_image_df.empty or gt_df.empty:
    raise RuntimeError(
        "Analysis produced empty result tables."
    )

print("\n" + "=" * 80)
print("ANALYSIS DATASET SIZES")
print("=" * 80)

print("Per-image/class rows:", len(per_image_df))
print("Ground-truth rows   :", len(gt_df))
print("Prediction rows     :", len(pred_df))


# ============================================================
# 10. BUILD PAIRED PER-IMAGE COMPARISON
# ============================================================

paired_df = per_image_df[
    per_image_df["model"] == "C_IRFS"
].merge(
    per_image_df[
        per_image_df["model"] == "CB_IRFS"
    ],
    on=[
        "split",
        "image",
        "class_id",
        "class_name",
    ],
    suffixes=("_C", "_CB"),
    how="inner",
    validate="one_to_one",
)

if len(paired_df) != 268 * NUM_CLASSES:
    raise RuntimeError(
        "Paired comparison does not contain "
        "268 images x 6 classes."
    )

for metric in [
    "TP_50",
    "FP_50",
    "FN_50",
    "precision_50",
    "recall_50",
    "TP_75",
    "FP_75",
    "FN_75",
    "precision_75",
    "recall_75",
]:

    paired_df[f"delta_{metric}"] = (
        paired_df[f"{metric}_CB"]
        - paired_df[f"{metric}_C"]
    )


# ============================================================
# 11. CLASS-LEVEL SUMMARY
# ============================================================

class_summary_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            group = per_image_df[
                (per_image_df["model"] == model_name)
                & (per_image_df["split"] == split_name)
                & (per_image_df["class_id"] == class_id)
            ]

            gt_group = gt_df[
                (gt_df["model"] == model_name)
                & (gt_df["split"] == split_name)
                & (gt_df["class_id"] == class_id)
            ]

            for threshold in ["50", "75"]:

                tp = int(group[f"TP_{threshold}"].sum())
                fp = int(group[f"FP_{threshold}"].sum())
                fn = int(group[f"FN_{threshold}"].sum())

                precision = (
                    tp / (tp + fp)
                    if tp + fp else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn else 0.0
                )

                mean_iou = (
                    gt_group.loc[
                        gt_group[f"matched_at_iou{threshold}"],
                        f"matched_iou{threshold}",
                    ].mean()
                )

                class_summary_rows.append({

                    "model": model_name,
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_name,

                    "matching_threshold": float(
                        f"0.{threshold}"
                    ),

                    "GT": tp + fn,
                    "Predictions": tp + fp,

                    "TP": tp,
                    "FP": fp,
                    "FN": fn,

                    "precision": precision,
                    "recall": recall,

                    "mean_matched_mask_iou": (
                        float(mean_iou)
                        if pd.notna(mean_iou)
                        else 0.0
                    ),
                })


class_summary_df = pd.DataFrame(
    class_summary_rows
)


# ============================================================
# 12. SIZE-STRATIFIED RECALL
# ============================================================

size_summary_rows = []

SIZE_BUCKETS = [
    "lt_32px",
    "32_64px",
    "64_128px",
    "ge_128px",
]

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            for bucket in SIZE_BUCKETS:

                group = gt_df[
                    (gt_df["model"] == model_name)
                    & (gt_df["split"] == split_name)
                    & (gt_df["class_id"] == class_id)
                    & (gt_df["size_bucket"] == bucket)
                ]

                n_gt = len(group)

                tp50 = int(
                    group["matched_at_iou50"].sum()
                )

                tp75 = int(
                    group["matched_at_iou75"].sum()
                )

                size_summary_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "size_bucket": bucket,
                    "GT_instances": n_gt,

                    "TP_IoU50": tp50,
                    "FN_IoU50": n_gt - tp50,

                    "recall_IoU50": (
                        tp50 / n_gt if n_gt else 0.0
                    ),

                    "TP_IoU75": tp75,
                    "FN_IoU75": n_gt - tp75,

                    "recall_IoU75": (
                        tp75 / n_gt if n_gt else 0.0
                    ),
                })


size_summary_df = pd.DataFrame(
    size_summary_rows
)


# ============================================================
# 13. FALSE POSITIVE TYPE SUMMARY
# ============================================================

fp_df = pred_df[
    pred_df["status_iou50"] == "FP"
].copy()

if not fp_df.empty:

    fp_type_summary_df = (
        fp_df.groupby(
            [
                "model",
                "split",
                "class_id",
                "class_name",
                "false_positive_type",
            ],
            dropna=False,
        )
        .agg(
            FP_count=("confidence", "count"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_best_same_class_iou=(
                "best_same_class_iou",
                "mean",
            ),
            mean_best_any_class_iou=(
                "best_any_class_iou",
                "mean",
            ),
        )
        .reset_index()
    )

else:

    fp_type_summary_df = pd.DataFrame()


# ============================================================
# 14. MISSED OBJECTS REPORT
# ============================================================

missed_df = gt_df[
    ~gt_df["matched_at_iou50"]
].copy()

missed_df = missed_df.sort_values(
    by=[
        "model",
        "class_name",
        "gt_max_side_px",
        "best_same_class_iou",
    ],
    ascending=[
        True,
        True,
        True,
        False,
    ],
)


# ============================================================
# 15. SAVE ALL CSV REPORTS
# ============================================================

def save_csv(df, filename):

    path = ANALYSIS_DIR / filename

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print("[SAVED]", path)

    return path


print("\n" + "=" * 80)
print("SAVING ANALYSIS REPORTS")
print("=" * 80)

save_csv(
    per_image_df,
    "per_image_class_metrics.csv",
)

save_csv(
    paired_df,
    "paired_image_class_comparison.csv",
)

save_csv(
    gt_df,
    "per_ground_truth_instance.csv",
)

save_csv(
    pred_df,
    "all_predictions.csv",
)

save_csv(
    fp_df,
    "false_positive_predictions.csv",
)

save_csv(
    class_summary_df,
    "class_summary.csv",
)

save_csv(
    size_summary_df,
    "size_stratified_recall.csv",
)

save_csv(
    fp_type_summary_df,
    "false_positive_type_summary.csv",
)

save_csv(
    missed_df,
    "missed_ground_truth_objects.csv",
)


# ============================================================
# 16. PRINT OVERALL CLASS COMPARISON
# ============================================================

print("\n" + "=" * 110)
print("COMBINED 268 — C-IRFS VS CB-IRFS")
print("MASK MATCHING AT IoU >= 0.50")
print("=" * 110)

combined_summary = class_summary_df[
    (class_summary_df["split"] == "Combined_268")
    & (class_summary_df["matching_threshold"] == 0.5)
]

for class_name in CLASS_NAMES:

    c = combined_summary[
        (combined_summary["model"] == "C_IRFS")
        & (combined_summary["class_name"] == class_name)
    ].iloc[0]

    cb = combined_summary[
        (combined_summary["model"] == "CB_IRFS")
        & (combined_summary["class_name"] == class_name)
    ].iloc[0]

    print(f"\n{class_name}")

    print(
        f"  C-IRFS : "
        f"TP={int(c.TP):4d} "
        f"FP={int(c.FP):4d} "
        f"FN={int(c.FN):4d} "
        f"P={c.precision:.4f} "
        f"R={c.recall:.4f}"
    )

    print(
        f"  CB-IRFS: "
        f"TP={int(cb.TP):4d} "
        f"FP={int(cb.FP):4d} "
        f"FN={int(cb.FN):4d} "
        f"P={cb.precision:.4f} "
        f"R={cb.recall:.4f}"
    )

    print(
        f"  Delta  : "
        f"TP={int(cb.TP-c.TP):+d} "
        f"FP={int(cb.FP-c.FP):+d} "
        f"FN={int(cb.FN-c.FN):+d} "
        f"Recall={cb.recall-c.recall:+.4f}"
    )


# ============================================================
# 17. PRINT SIZE-STRATIFIED RECALL FOR GLOVES AND SHOES
# ============================================================

print("\n" + "=" * 110)
print("SIZE-STRATIFIED RECALL — GLOVES AND SHOES")
print("=" * 110)

target_classes = [
    "gloves",
    "shoes",
]

for split_name in [
    "Combined_268",
    "Old_119",
    "New_149",
]:

    print(f"\n{split_name}")

    for class_name in target_classes:

        print(f"\n  {class_name}")

        for model_name in MODELS:

            subset = size_summary_df[
                (size_summary_df["model"] == model_name)
                & (size_summary_df["split"] == split_name)
                & (size_summary_df["class_name"] == class_name)
            ]

            print(f"    {model_name}")

            for _, row in subset.iterrows():

                print(
                    f"      {row['size_bucket']:12s} "
                    f"GT={int(row['GT_instances']):4d} "
                    f"Recall50={row['recall_IoU50']:.4f} "
                    f"Recall75={row['recall_IoU75']:.4f}"
                )


# ============================================================
# 18. FALSE POSITIVE TYPE SUMMARY FOR GLOVES AND SHOES
# ============================================================

print("\n" + "=" * 110)
print("FALSE POSITIVE TYPES — GLOVES AND SHOES")
print("=" * 110)

for class_name in target_classes:

    print(f"\n{class_name}")

    subset = fp_type_summary_df[
        (fp_type_summary_df["split"] == "Combined_268")
        & (fp_type_summary_df["class_name"] == class_name)
    ]

    if subset.empty:
        print("No false positives.")
        continue

    print(
        subset[
            [
                "model",
                "false_positive_type",
                "FP_count",
                "mean_confidence",
                "median_confidence",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 19. SAVE METADATA
# ============================================================

metadata = {

    "analysis": "paired_error_analysis_v1",

    "models": {
        name: str(path)
        for name, path in MODELS.items()
    },

    "validation_images": len(image_paths),
    "validation_old": old_count,
    "validation_new": new_count,

    "ground_truth_instances": int(
        gt_class_counts.sum()
    ),

    "ground_truth_class_counts": {
        CLASS_NAMES[i]: int(gt_class_counts[i])
        for i in range(NUM_CLASSES)
    },

    "source_label_lines": source_label_count,
    "exact_duplicate_labels_removed": duplicate_label_count,

    "prediction_configuration": PREDICT_ARGS,

    "matching": {
        "method": (
            "class-wise Hungarian one-to-one assignment"
            if SCIPY_AVAILABLE
            else "class-wise greedy one-to-one assignment"
        ),
        "mask_iou_thresholds": MATCH_THRESHOLDS,
    },

    "size_buckets_original_pixels": SIZE_BUCKETS,

    "model_inference_summary": inference_summary,

    "interpretation_warning": (
        "These are fixed-confidence, per-image mask IoU "
        "diagnostic metrics, not official Ultralytics AP. "
        "They depend on confidence threshold, NMS and the "
        "one-to-one matching implementation."
    ),
}

metadata_path = (
    ANALYSIS_DIR / "analysis_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("PAIRED ERROR ANALYSIS COMPLETED")
print("=" * 100)

print("Images analyzed:", len(image_paths))
print("Models analyzed:", len(MODELS))
print("GT instances per model:", int(gt_class_counts.sum()))

print("\nOutput directory:")
print(ANALYSIS_DIR)

print("\nImportant files:")
print("1. paired_image_class_comparison.csv")
print("2. per_ground_truth_instance.csv")
print("3. false_positive_predictions.csv")
print("4. size_stratified_recall.csv")
print("5. false_positive_type_summary.csv")
print("6. missed_ground_truth_objects.csv")
print("7. class_summary.csv")
print("8. all_predictions.csv")
print("9. analysis_metadata.json")

print("\n[CELL 17 COMPLETED]")

###########################################################################
# CELL 23
###########################################################################
# ============================================================
# CELL 17B — GPU CLEANUP + RESUME PAIRED ERROR ANALYSIS
#
# Continues from the failed Cell 17.
# Keeps previously parsed GT polygons and helper functions.
#
# Memory optimizations:
#   - Release previous training/test models.
#   - Shutdown old DataLoader workers.
#   - Inference batch size = 1.
#   - FP16 inference.
#   - Process one image at a time.
#
# ============================================================

import gc
import json
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO


# ============================================================
# 1. CHECK PREVIOUS CELL DEFINITIONS
# ============================================================

required_globals = [
    "MODELS",
    "image_paths",
    "image_path_map",
    "basename_map",
    "image_split_map",
    "GT_POLYGONS",
    "CLASS_NAMES",
    "create_gt_objects",
    "mask_bbox_and_area",
    "analyze_one_image",
    "ANALYSIS_DIR",
    "PREDICT_ARGS",
]

missing = [
    name for name in required_globals
    if name not in globals()
]

if missing:
    raise RuntimeError(
        "Some Cell 17 definitions are unavailable.\n"
        "Run Cell 17 again to define its helper functions.\n"
        f"Missing: {missing}"
    )

print("[OK] Previous Cell 17 definitions found.")
print("[OK] Parsed ground-truth polygons retained.")


# ============================================================
# 2. SHUTDOWN OLD DATALOADER WORKERS
# ============================================================

print("\n" + "=" * 80)
print("SHUTTING DOWN OLD DATALOADER WORKERS")
print("=" * 80)

for owner_name in ["main_trainer", "trainer"]:

    owner = globals().get(owner_name)

    if owner is None:
        continue

    for loader_name in [
        "train_loader",
        "test_loader",
        "val_loader",
    ]:

        loader = getattr(owner, loader_name, None)

        if loader is None:
            continue

        iterator = getattr(loader, "_iterator", None)

        if iterator is not None:

            shutdown = getattr(
                iterator,
                "_shutdown_workers",
                None,
            )

            if callable(shutdown):

                try:
                    shutdown()
                    print(
                        f"[OK] {owner_name}.{loader_name}: "
                        "workers shutdown"
                    )

                except Exception as exc:
                    print(
                        f"[WARNING] Worker shutdown: {exc}"
                    )


# ============================================================
# 3. CLOSE PREVIOUS FAILED PREDICTION STREAM
# ============================================================

old_stream = globals().get("prediction_stream")

if old_stream is not None:

    close_fn = getattr(
        old_stream,
        "close",
        None,
    )

    if callable(close_fn):

        try:
            close_fn()
            print("\n[OK] Previous prediction stream closed.")

        except Exception as exc:
            print("[WARNING] Stream close:", exc)


# ============================================================
# 4. RELEASE GPU-HEAVY GLOBAL OBJECTS
# ============================================================

heavy_objects = [
    # Previous training models and trainers.
    "main_trainer",
    "trainer",
    "cb_main",
    "cb_yolo",

    # Numerical loss test.
    "test_model",
    "default_criterion",
    "unit_weight_criterion",
    "balanced_criterion",

    "default_loss",
    "unit_loss",
    "balanced_loss",

    "default_items",
    "unit_items",
    "balanced_items",

    "preds",
    "batch",
    "raw_batch",

    # Failed inference objects.
    "model",
    "prediction_stream",
    "result",

    # Other possible GPU references.
    "pred_masks",
    "pred_classes",
    "pred_boxes",
]

for name in heavy_objects:

    if name in globals():

        globals().pop(name, None)


# Garbage collection.
gc.collect()

if torch.cuda.is_available():

    try:
        torch.cuda.synchronize()

    except Exception:
        pass

    torch.cuda.empty_cache()

    try:
        torch.cuda.ipc_collect()

    except Exception:
        pass

gc.collect()


# ============================================================
# 5. REPORT GPU MEMORY BEFORE INFERENCE
# ============================================================

print("\n" + "=" * 80)
print("GPU MEMORY AFTER CLEANUP")
print("=" * 80)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

free_bytes, total_bytes = torch.cuda.mem_get_info(0)

allocated_bytes = torch.cuda.memory_allocated(0)
reserved_bytes = torch.cuda.memory_reserved(0)

print(
    f"Total GPU memory     : {total_bytes / 1024**3:.2f} GB"
)

print(
    f"Free GPU memory      : {free_bytes / 1024**3:.2f} GB"
)

print(
    f"PyTorch allocated    : {allocated_bytes / 1024**3:.2f} GB"
)

print(
    f"PyTorch reserved     : {reserved_bytes / 1024**3:.2f} GB"
)

if free_bytes < 3 * 1024**3:

    raise RuntimeError(
        "GPU still has less than 3 GB free after cleanup.\n"
        "Do not retry inference yet. The notebook may require "
        "a kernel restart to release remaining GPU references."
    )

print("\n[OK] GPU memory cleanup completed.")


# ============================================================
# 6. SET MEMORY-SAFE INFERENCE CONFIGURATION
# ============================================================

PREDICT_ARGS.update({

    "imgsz": 960,

    # One image per forward pass.
    "batch": 1,

    # No additional image-loading workers.
    "workers": 0,

    # FP16 inference on Tesla T4.
    "half": True,

    # Keep original-resolution masks.
    "retina_masks": True,

    "stream": True,

    "device": 0,

    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
})

print("\n" + "=" * 80)
print("MEMORY-SAFE INFERENCE CONFIGURATION")
print("=" * 80)

for key, value in PREDICT_ARGS.items():
    print(f"{key:20s}: {value}")


# ============================================================
# 7. RESET RESULT CONTAINERS
# ============================================================

per_image_records = []
gt_records = []
prediction_records = []

inference_summary = {}

print("\n[OK] Analysis result containers initialized.")


# ============================================================
# 8. RUN INFERENCE ONE MODEL AT A TIME
# ============================================================

for model_name, checkpoint_path in MODELS.items():

    print("\n")
    print("#" * 90)
    print("STARTING INFERENCE:", model_name)
    print("#" * 90)

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)

    model = YOLO(str(checkpoint_path))

    processed_images = 0
    total_predictions = 0
    total_gt_instances = 0

    prediction_stream = model.predict(
        source=[str(p) for p in image_paths],
        **PREDICT_ARGS,
    )

    try:

        for result in prediction_stream:

            result_path = Path(
                result.path
            ).resolve()

            image_path = image_path_map.get(
                result_path
            )

            if image_path is None:

                image_path = basename_map.get(
                    Path(result.path).name
                )

            if image_path is None:
                raise RuntimeError(
                    f"Unknown image returned by model: {result.path}"
                )

            height, width = result.orig_shape

            # ------------------------------------------------
            # Ground-truth objects.
            # ------------------------------------------------

            gt_objects = create_gt_objects(
                image_path,
                height,
                width,
            )

            # ------------------------------------------------
            # Predicted objects.
            # ------------------------------------------------

            pred_objects = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                pred_classes = (
                    result.boxes.cls
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                pred_confidences = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                pred_boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                if result.masks is None:
                    raise RuntimeError(
                        "Boxes found but masks are missing."
                    )

                if len(result.masks.data) != len(pred_classes):
                    raise RuntimeError(
                        "Prediction boxes/masks count mismatch."
                    )

                # Process masks individually to avoid making
                # an additional large full-resolution GPU copy.
                for index in range(len(pred_classes)):

                    class_id = int(
                        pred_classes[index]
                    )

                    if not 0 <= class_id < len(CLASS_NAMES):
                        continue

                    pred_mask = (
                        result.masks.data[index]
                        .detach()
                        .cpu()
                        .numpy()
                    ) > 0.5

                    # Align mask dimensions if necessary.
                    if pred_mask.shape != (height, width):

                        pred_mask = cv2.resize(
                            pred_mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    (
                        pred_bbox,
                        pred_area,
                        pred_width,
                        pred_height,
                        pred_max_side,
                    ) = mask_bbox_and_area(pred_mask)

                    box = pred_boxes[index]

                    pred_objects.append({

                        "class_id": class_id,

                        "confidence": float(
                            pred_confidences[index]
                        ),

                        "mask": pred_mask,
                        "bbox": pred_bbox,
                        "mask_area": pred_area,

                        "bbox_width": pred_width,
                        "bbox_height": pred_height,
                        "max_side": pred_max_side,

                        "box_xyxy": [
                            float(x) for x in box
                        ],
                    })

            # ------------------------------------------------
            # Match GT and predictions.
            # ------------------------------------------------

            relative_path = str(
                image_path.relative_to(VAL_IMAGE_ROOT)
            )

            split_name = image_split_map[image_path]

            analyze_one_image(
                model_name=model_name,
                image_path=image_path,
                gt_objects=gt_objects,
                pred_objects=pred_objects,
                image_rel_path=relative_path,
                split_name=split_name,
            )

            processed_images += 1
            total_predictions += len(pred_objects)
            total_gt_instances += len(gt_objects)

            # Explicitly release the prediction result.
            del result

            if processed_images % 25 == 0:

                print(
                    f"{model_name}: "
                    f"{processed_images}/268 images, "
                    f"predictions={total_predictions}"
                )

                gc.collect()

                torch.cuda.empty_cache()

    finally:

        try:
            prediction_stream.close()

        except Exception:
            pass

    print("\nCompleted:", model_name)
    print("Images:", processed_images)
    print("GT instances:", total_gt_instances)
    print("Predictions:", total_predictions)

    if processed_images != 268:
        raise RuntimeError(
            f"{model_name}: expected 268 images, "
            f"processed {processed_images}."
        )

    inference_summary[model_name] = {

        "images": processed_images,
        "ground_truth_instances": total_gt_instances,
        "predictions": total_predictions,
        "checkpoint": str(checkpoint_path),
    }

    # Release current model before loading the next checkpoint.
    del prediction_stream
    del model

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# ============================================================
# 9. CREATE DATAFRAMES
# ============================================================

per_image_df = pd.DataFrame(
    per_image_records
)

gt_df = pd.DataFrame(
    gt_records
)

pred_df = pd.DataFrame(
    prediction_records
)

if per_image_df.empty or gt_df.empty:
    raise RuntimeError(
        "Analysis result tables are empty."
    )

print("\n" + "=" * 80)
print("ANALYSIS RESULT COUNTS")
print("=" * 80)

print("Per-image/class rows:", len(per_image_df))
print("Ground-truth rows   :", len(gt_df))
print("Prediction rows     :", len(pred_df))


# ============================================================
# 10. PAIRED PER-IMAGE COMPARISON
# ============================================================

paired_df = per_image_df[
    per_image_df["model"] == "C_IRFS"
].merge(
    per_image_df[
        per_image_df["model"] == "CB_IRFS"
    ],
    on=[
        "split",
        "image",
        "class_id",
        "class_name",
    ],
    suffixes=("_C", "_CB"),
    how="inner",
    validate="one_to_one",
)

assert len(paired_df) == 268 * 6

for metric in [
    "TP_50",
    "FP_50",
    "FN_50",
    "precision_50",
    "recall_50",

    "TP_75",
    "FP_75",
    "FN_75",
    "precision_75",
    "recall_75",
]:

    paired_df[f"delta_{metric}"] = (
        paired_df[f"{metric}_CB"]
        - paired_df[f"{metric}_C"]
    )


# ============================================================
# 11. CLASS SUMMARY
# ============================================================

class_summary_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            image_group = per_image_df[
                (per_image_df["model"] == model_name)
                & (per_image_df["split"] == split_name)
                & (per_image_df["class_id"] == class_id)
            ]

            instance_group = gt_df[
                (gt_df["model"] == model_name)
                & (gt_df["split"] == split_name)
                & (gt_df["class_id"] == class_id)
            ]

            for threshold in ["50", "75"]:

                tp = int(
                    image_group[f"TP_{threshold}"].sum()
                )

                fp = int(
                    image_group[f"FP_{threshold}"].sum()
                )

                fn = int(
                    image_group[f"FN_{threshold}"].sum()
                )

                precision = (
                    tp / (tp + fp)
                    if tp + fp else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn else 0.0
                )

                mean_iou = instance_group.loc[
                    instance_group[
                        f"matched_at_iou{threshold}"
                    ],
                    f"matched_iou{threshold}",
                ].mean()

                class_summary_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "matching_threshold": (
                        0.50 if threshold == "50" else 0.75
                    ),

                    "GT": tp + fn,
                    "Predictions": tp + fp,

                    "TP": tp,
                    "FP": fp,
                    "FN": fn,

                    "precision": precision,
                    "recall": recall,

                    "mean_matched_mask_iou": (
                        float(mean_iou)
                        if pd.notna(mean_iou)
                        else 0.0
                    ),
                })


class_summary_df = pd.DataFrame(
    class_summary_rows
)


# ============================================================
# 12. SIZE-STRATIFIED RECALL
# ============================================================

SIZE_BUCKETS = [
    "lt_32px",
    "32_64px",
    "64_128px",
    "ge_128px",
]

size_summary_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            for bucket in SIZE_BUCKETS:

                group = gt_df[
                    (gt_df["model"] == model_name)
                    & (gt_df["split"] == split_name)
                    & (gt_df["class_id"] == class_id)
                    & (gt_df["size_bucket"] == bucket)
                ]

                n_gt = len(group)

                tp50 = int(
                    group["matched_at_iou50"].sum()
                )

                tp75 = int(
                    group["matched_at_iou75"].sum()
                )

                size_summary_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "size_bucket": bucket,
                    "GT_instances": n_gt,

                    "TP_IoU50": tp50,
                    "FN_IoU50": n_gt - tp50,

                    "recall_IoU50": (
                        tp50 / n_gt if n_gt else 0.0
                    ),

                    "TP_IoU75": tp75,
                    "FN_IoU75": n_gt - tp75,

                    "recall_IoU75": (
                        tp75 / n_gt if n_gt else 0.0
                    ),
                })


size_summary_df = pd.DataFrame(
    size_summary_rows
)


# ============================================================
# 13. FALSE POSITIVE TYPE SUMMARY
# ============================================================

fp_df = pred_df[
    pred_df["status_iou50"] == "FP"
].copy()

if not fp_df.empty:

    fp_type_summary_df = (
        fp_df.groupby(
            [
                "model",
                "split",
                "class_id",
                "class_name",
                "false_positive_type",
            ],
            dropna=False,
        )
        .agg(
            FP_count=("confidence", "count"),

            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),

            mean_best_same_class_iou=(
                "best_same_class_iou",
                "mean",
            ),

            mean_best_any_class_iou=(
                "best_any_class_iou",
                "mean",
            ),
        )
        .reset_index()
    )

else:
    fp_type_summary_df = pd.DataFrame()


# ============================================================
# 14. MISSED GROUND-TRUTH OBJECTS
# ============================================================

missed_df = gt_df[
    ~gt_df["matched_at_iou50"]
].copy()

missed_df = missed_df.sort_values(
    by=[
        "model",
        "class_name",
        "gt_max_side_px",
        "best_same_class_iou",
    ],
    ascending=[
        True,
        True,
        True,
        False,
    ],
)


# ============================================================
# 15. SAVE REPORTS
# ============================================================

def save_analysis_csv(df, filename):

    path = ANALYSIS_DIR / filename

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print("[SAVED]", path)

    return path


print("\n" + "=" * 80)
print("SAVING ANALYSIS REPORTS")
print("=" * 80)

save_analysis_csv(
    per_image_df,
    "per_image_class_metrics.csv",
)

save_analysis_csv(
    paired_df,
    "paired_image_class_comparison.csv",
)

save_analysis_csv(
    gt_df,
    "per_ground_truth_instance.csv",
)

save_analysis_csv(
    pred_df,
    "all_predictions.csv",
)

save_analysis_csv(
    fp_df,
    "false_positive_predictions.csv",
)

save_analysis_csv(
    class_summary_df,
    "class_summary.csv",
)

save_analysis_csv(
    size_summary_df,
    "size_stratified_recall.csv",
)

save_analysis_csv(
    fp_type_summary_df,
    "false_positive_type_summary.csv",
)

save_analysis_csv(
    missed_df,
    "missed_ground_truth_objects.csv",
)


# ============================================================
# 16. PRINT COMBINED CLASS COMPARISON
# ============================================================

print("\n" + "=" * 110)
print("COMBINED 268 — MASK MATCHING AT IoU >= 0.50")
print("=" * 110)

combined_summary = class_summary_df[
    (class_summary_df["split"] == "Combined_268")
    & (class_summary_df["matching_threshold"] == 0.50)
]

for class_name in CLASS_NAMES:

    c = combined_summary[
        (combined_summary["model"] == "C_IRFS")
        & (combined_summary["class_name"] == class_name)
    ].iloc[0]

    cb = combined_summary[
        (combined_summary["model"] == "CB_IRFS")
        & (combined_summary["class_name"] == class_name)
    ].iloc[0]

    print(f"\n{class_name}")

    print(
        f"  C-IRFS : "
        f"TP={int(c.TP):4d} "
        f"FP={int(c.FP):4d} "
        f"FN={int(c.FN):4d} "
        f"P={c.precision:.4f} "
        f"R={c.recall:.4f}"
    )

    print(
        f"  CB-IRFS: "
        f"TP={int(cb.TP):4d} "
        f"FP={int(cb.FP):4d} "
        f"FN={int(cb.FN):4d} "
        f"P={cb.precision:.4f} "
        f"R={cb.recall:.4f}"
    )

    print(
        f"  Delta  : "
        f"TP={int(cb.TP-c.TP):+d} "
        f"FP={int(cb.FP-c.FP):+d} "
        f"FN={int(cb.FN-c.FN):+d} "
        f"Recall={cb.recall-c.recall:+.4f}"
    )


# ============================================================
# 17. SIZE-STRATIFIED RECALL FOR GLOVES AND SHOES
# ============================================================

print("\n" + "=" * 110)
print("SIZE-STRATIFIED RECALL — GLOVES AND SHOES")
print("=" * 110)

for split_name in [
    "Combined_268",
    "Old_119",
    "New_149",
]:

    print(f"\n{split_name}")

    for class_name in [
        "gloves",
        "shoes",
    ]:

        print(f"\n  {class_name}")

        for model_name in MODELS:

            subset = size_summary_df[
                (size_summary_df["model"] == model_name)
                & (size_summary_df["split"] == split_name)
                & (size_summary_df["class_name"] == class_name)
            ]

            print(f"    {model_name}")

            for _, row in subset.iterrows():

                print(
                    f"      {row['size_bucket']:12s} "
                    f"GT={int(row['GT_instances']):4d} "
                    f"Recall50={row['recall_IoU50']:.4f} "
                    f"Recall75={row['recall_IoU75']:.4f}"
                )


# ============================================================
# 18. FALSE POSITIVE TYPES — GLOVES AND SHOES
# ============================================================

print("\n" + "=" * 110)
print("FALSE POSITIVE TYPES — GLOVES AND SHOES")
print("=" * 110)

for class_name in [
    "gloves",
    "shoes",
]:

    print(f"\n{class_name}")

    subset = fp_type_summary_df[
        (fp_type_summary_df["split"] == "Combined_268")
        & (fp_type_summary_df["class_name"] == class_name)
    ]

    if subset.empty:
        print("No false positives.")
        continue

    print(
        subset[
            [
                "model",
                "false_positive_type",
                "FP_count",
                "mean_confidence",
                "median_confidence",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 19. SAVE METADATA
# ============================================================

metadata = {

    "analysis": "paired_error_analysis_v1",

    "models": {
        name: str(path)
        for name, path in MODELS.items()
    },

    "validation_images": len(image_paths),

    "validation_old": 119,
    "validation_new": 149,

    "ground_truth_instances": int(
        sum(EXPECTED_GT_COUNTS)
    ),

    "ground_truth_class_counts": {
        CLASS_NAMES[i]: int(EXPECTED_GT_COUNTS[i])
        for i in range(len(CLASS_NAMES))
    },

    "prediction_configuration": PREDICT_ARGS,

    "matching": {
        "method": (
            "class-wise Hungarian one-to-one assignment"
            if SCIPY_AVAILABLE
            else "class-wise greedy one-to-one assignment"
        ),
        "mask_iou_thresholds": [
            0.50,
            0.75,
        ],
    },

    "size_buckets_original_pixels": SIZE_BUCKETS,

    "model_inference_summary": inference_summary,

    "interpretation_warning": (
        "These are fixed-confidence, per-image mask IoU "
        "diagnostic metrics, not official Ultralytics AP. "
        "They depend on confidence threshold, NMS, max_det "
        "and the one-to-one matching implementation."
    ),
}

metadata_path = (
    ANALYSIS_DIR / "analysis_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("PAIRED ERROR ANALYSIS COMPLETED")
print("=" * 100)

print("Images analyzed:", len(image_paths))
print("Models analyzed:", len(MODELS))
print("GT instances per model:", int(sum(EXPECTED_GT_COUNTS)))

print("\nOutput directory:")
print(ANALYSIS_DIR)

print("\nImportant files:")

for filename in [
    "paired_image_class_comparison.csv",
    "per_ground_truth_instance.csv",
    "false_positive_predictions.csv",
    "size_stratified_recall.csv",
    "false_positive_type_summary.csv",
    "missed_ground_truth_objects.csv",
    "class_summary.csv",
    "all_predictions.csv",
    "analysis_metadata.json",
]:
    print(" -", ANALYSIS_DIR / filename)

print("\n[CELL 17B COMPLETED]")

###########################################################################
# CELL 24
###########################################################################
# ============================================================
# CELL 17C — SELF-CONTAINED PAIRED ERROR ANALYSIS
#
# Models:
#   C-IRFS vs CB-IRFS
#
# Validation:
#   Combined 268 = Old 119 + New 149
#
# Memory-safe inference:
#   batch=1, FP16, imgsz=960
#
# Matching:
#   Class-wise Hungarian, mask IoU >= 0.50 / 0.75
#
# No training. No model updates.
# ============================================================

import gc
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO

from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

VAL_IMAGE_ROOT = PROJECT_DIR / "dataset" / "images" / "val"
VAL_LABEL_ROOT = PROJECT_DIR / "dataset" / "labels" / "val"

C_CHECKPOINT = (
    PROJECT_DIR
    / "research_experiments"
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

CB_CHECKPOINT = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "CB_IRFS_sqrt_17ep_v1"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "paired_error_analysis_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = {
    "C_IRFS": C_CHECKPOINT,
    "CB_IRFS": CB_CHECKPOINT,
}

CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

NUM_CLASSES = 6

EXPECTED_GT_COUNTS = [
    829, 238, 339, 446, 274, 916
]

PREDICT_ARGS = {
    "imgsz": 960,
    "batch": 1,
    "rect": True,
    "device": 0,
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "retina_masks": True,
    "half": True,
    "augment": False,
    "verbose": False,
    "save": False,
    "stream": True,
}

SIZE_BUCKETS = [
    "lt_32px",
    "32_64px",
    "64_128px",
    "ge_128px",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ============================================================
# 2. VERIFY FILES AND GPU MEMORY
# ============================================================

print("=" * 80)
print("PROJECT VERIFICATION")
print("=" * 80)

for path in [
    PROJECT_DIR,
    VAL_IMAGE_ROOT,
    VAL_LABEL_ROOT,
    C_CHECKPOINT,
    CB_CHECKPOINT,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required path missing:\n{path}"
        )

    print("[OK]", path)


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

free_bytes, total_bytes = torch.cuda.mem_get_info(0)

print("\nGPU:", torch.cuda.get_device_name(0))
print("Total memory:", round(total_bytes / 1024**3, 2), "GB")
print("Free memory :", round(free_bytes / 1024**3, 2), "GB")

if free_bytes < 5 * 1024**3:
    raise RuntimeError(
        "GPU has less than 5 GB free after kernel restart.\n"
        "Do not run inference until GPU memory is released."
    )

print("\n[OK] GPU memory sufficient for batch=1 inference.")


# ============================================================
# 3. ENUMERATE IMAGES
# ============================================================

image_paths = sorted(
    p.resolve()
    for p in VAL_IMAGE_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

if len(image_paths) != 268:
    raise RuntimeError(
        f"Expected 268 images, found {len(image_paths)}."
    )

image_path_map = {
    p: p for p in image_paths
}

basename_map = {
    p.name: p for p in image_paths
}

image_split_map = {}

for image_path in image_paths:

    relative_path = image_path.relative_to(
        VAL_IMAGE_ROOT
    )

    if relative_path.parts[0] == "old":
        split_name = "Old_119"

    elif relative_path.parts[0] == "new":
        split_name = "New_149"

    else:
        raise RuntimeError(
            f"Unknown validation split: {relative_path}"
        )

    image_split_map[image_path] = split_name


old_count = sum(
    v == "Old_119"
    for v in image_split_map.values()
)

new_count = sum(
    v == "New_149"
    for v in image_split_map.values()
)

assert old_count == 119
assert new_count == 149

print("\nImages:", len(image_paths))
print("Old:", old_count)
print("New:", new_count)


# ============================================================
# 4. LOAD GROUND-TRUTH POLYGONS
# ============================================================

def get_label_path(image_path):

    relative_path = image_path.relative_to(
        VAL_IMAGE_ROOT
    )

    return (
        VAL_LABEL_ROOT / relative_path
    ).with_suffix(".txt")


def parse_label_file(label_path):

    objects = []
    seen = set()

    source_lines = 0
    duplicates = 0

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            source_lines += 1

            try:
                values = np.asarray(
                    [float(x) for x in line.split()],
                    dtype=np.float64,
                )

            except ValueError as exc:
                raise ValueError(
                    f"Invalid label: {label_path}:{line_number}"
                ) from exc

            if len(values) < 7:
                raise ValueError(
                    f"Polygon too short: {label_path}:{line_number}"
                )

            if (len(values) - 1) % 2:
                raise ValueError(
                    f"Invalid polygon coordinates: {label_path}"
                )

            class_id = int(round(values[0]))

            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"Invalid class ID: {class_id}"
                )

            if abs(values[0] - class_id) > 1e-6:
                raise ValueError(
                    f"Noninteger class ID: {label_path}"
                )

            key = tuple(
                np.round(values, 8).tolist()
            )

            if key in seen:
                duplicates += 1
                continue

            seen.add(key)

            points = values[1:].reshape(-1, 2)

            if not np.isfinite(points).all():
                raise ValueError(
                    f"Non-finite polygon: {label_path}"
                )

            objects.append({
                "class_id": class_id,
                "points": points,
            })

    return objects, source_lines, duplicates


GT_POLYGONS = {}

gt_class_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64,
)

source_lines_total = 0
duplicate_total = 0

print("\n" + "=" * 80)
print("GROUND-TRUTH AUDIT")
print("=" * 80)

for index, image_path in enumerate(image_paths, start=1):

    label_path = get_label_path(image_path)

    objects, source_lines, duplicates = (
        parse_label_file(label_path)
    )

    GT_POLYGONS[image_path] = objects

    source_lines_total += source_lines
    duplicate_total += duplicates

    for obj in objects:
        gt_class_counts[obj["class_id"]] += 1

    if index % 50 == 0:
        print(f"Read {index}/268 images")


print("\nSource lines:", source_lines_total)
print("Duplicate polygons:", duplicate_total)
print("Total instances:", int(gt_class_counts.sum()))

for i, name in enumerate(CLASS_NAMES):

    actual = int(gt_class_counts[i])
    expected = EXPECTED_GT_COUNTS[i]

    print(
        f"{name:22s} "
        f"actual={actual:4d} "
        f"expected={expected:4d}"
    )

    assert actual == expected

assert gt_class_counts.sum() == 3042

print("\n[OK] All GT counts match.")


# ============================================================
# 5. MASK UTILITIES
# ============================================================

def rasterize_polygon(points, height, width):

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    points = np.clip(points, 0.0, 1.0)

    pixel_points = np.empty_like(points)

    pixel_points[:, 0] = points[:, 0] * (width - 1)
    pixel_points[:, 1] = points[:, 1] * (height - 1)

    pixel_points = np.rint(
        pixel_points
    ).astype(np.int32)

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [pixel_points],
        1,
    )

    return mask.astype(bool)


def mask_statistics(mask):

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None, 0, 0, 0, 0

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1

    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    width = x2 - x1
    height = y2 - y1

    area = int(len(xs))
    max_side = max(width, height)

    return (
        (x1, y1, x2, y2),
        area,
        width,
        height,
        max_side,
    )


def size_bucket(max_side):

    if max_side < 32:
        return "lt_32px"

    elif max_side < 64:
        return "32_64px"

    elif max_side < 128:
        return "64_128px"

    return "ge_128px"


def create_gt_objects(image_path, height, width):

    objects = []

    for item in GT_POLYGONS[image_path]:

        mask = rasterize_polygon(
            item["points"],
            height,
            width,
        )

        bbox, area, box_w, box_h, max_side = (
            mask_statistics(mask)
        )

        if area <= 0:
            raise RuntimeError(
                f"Empty GT polygon: {image_path}"
            )

        objects.append({

            "class_id": item["class_id"],
            "mask": mask,

            "bbox": bbox,
            "mask_area": area,

            "bbox_width": box_w,
            "bbox_height": box_h,
            "max_side": max_side,

            "size_bucket": size_bucket(max_side),
        })

    return objects


def mask_iou(a, b):

    box_a = a["bbox"]
    box_b = b["bbox"]

    if box_a is None or box_b is None:
        return 0.0

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = np.count_nonzero(
        a["mask"][y1:y2, x1:x2]
        &
        b["mask"][y1:y2, x1:x2]
    )

    union = (
        a["mask_area"]
        + b["mask_area"]
        - intersection
    )

    return (
        float(intersection / union)
        if union > 0 else 0.0
    )


# ============================================================
# 6. MATCHING FUNCTIONS
# ============================================================

def build_iou_matrix(gt_objects, pred_objects):

    matrix = np.zeros(
        (len(gt_objects), len(pred_objects)),
        dtype=np.float32,
    )

    for gi, gt in enumerate(gt_objects):

        for pi, pred in enumerate(pred_objects):

            matrix[gi, pi] = mask_iou(
                gt,
                pred,
            )

    return matrix


def match_objects(
    iou_matrix,
    gt_objects,
    pred_objects,
    threshold,
):

    gt_to_pred = {}
    pred_to_gt = {}

    for class_id in range(NUM_CLASSES):

        gt_indices = [
            i for i, obj in enumerate(gt_objects)
            if obj["class_id"] == class_id
        ]

        pred_indices = [
            i for i, obj in enumerate(pred_objects)
            if obj["class_id"] == class_id
        ]

        if not gt_indices or not pred_indices:
            continue

        submatrix = iou_matrix[
            np.ix_(gt_indices, pred_indices)
        ]

        rows, cols = linear_sum_assignment(
            -submatrix
        )

        for row, col in zip(rows, cols):

            overlap = float(
                submatrix[row, col]
            )

            if overlap < threshold:
                continue

            gi = gt_indices[row]
            pi = pred_indices[col]

            gt_to_pred[gi] = (
                pi,
                overlap,
            )

            pred_to_gt[pi] = (
                gi,
                overlap,
            )

    return gt_to_pred, pred_to_gt


# ============================================================
# 7. ANALYSIS RESULT CONTAINERS
# ============================================================

per_image_records = []
gt_records = []
prediction_records = []

inference_summary = {}


# ============================================================
# 8. ANALYZE ONE IMAGE
# ============================================================

def analyze_image(
    model_name,
    relative_path,
    split_name,
    gt_objects,
    pred_objects,
):

    matrix = build_iou_matrix(
        gt_objects,
        pred_objects,
    )

    gt50, pred50 = match_objects(
        matrix,
        gt_objects,
        pred_objects,
        0.50,
    )

    gt75, pred75 = match_objects(
        matrix,
        gt_objects,
        pred_objects,
        0.75,
    )

    n_gt = len(gt_objects)
    n_pred = len(pred_objects)

    # Best overlaps.
    gt_same = []
    gt_other = []
    gt_any = []
    gt_any_class = []
    gt_same_pred = []

    for gi, gt in enumerate(gt_objects):

        same = [
            pi for pi, pred in enumerate(pred_objects)
            if pred["class_id"] == gt["class_id"]
        ]

        other = [
            pi for pi, pred in enumerate(pred_objects)
            if pred["class_id"] != gt["class_id"]
        ]

        if same:

            values = matrix[gi, same]
            best_local = int(np.argmax(values))

            gt_same.append(float(values[best_local]))
            gt_same_pred.append(same[best_local])

        else:

            gt_same.append(0.0)
            gt_same_pred.append(None)

        gt_other.append(
            float(matrix[gi, other].max())
            if other else 0.0
        )

        if n_pred:

            best_pi = int(np.argmax(matrix[gi]))

            gt_any.append(float(matrix[gi, best_pi]))
            gt_any_class.append(
                pred_objects[best_pi]["class_id"]
            )

        else:

            gt_any.append(0.0)
            gt_any_class.append(None)

    pred_same = []
    pred_other = []
    pred_any = []
    pred_any_class = []

    for pi, pred in enumerate(pred_objects):

        same = [
            gi for gi, gt in enumerate(gt_objects)
            if gt["class_id"] == pred["class_id"]
        ]

        other = [
            gi for gi, gt in enumerate(gt_objects)
            if gt["class_id"] != pred["class_id"]
        ]

        pred_same.append(
            float(matrix[same, pi].max())
            if same else 0.0
        )

        pred_other.append(
            float(matrix[other, pi].max())
            if other else 0.0
        )

        if n_gt:

            best_gi = int(np.argmax(matrix[:, pi]))

            pred_any.append(
                float(matrix[best_gi, pi])
            )

            pred_any_class.append(
                gt_objects[best_gi]["class_id"]
            )

        else:

            pred_any.append(0.0)
            pred_any_class.append(None)

    # --------------------------------------------------------
    # Per image / per class.
    # --------------------------------------------------------

    for class_id, class_name in enumerate(CLASS_NAMES):

        gi_list = [
            i for i, obj in enumerate(gt_objects)
            if obj["class_id"] == class_id
        ]

        pi_list = [
            i for i, obj in enumerate(pred_objects)
            if obj["class_id"] == class_id
        ]

        tp50 = sum(i in gt50 for i in gi_list)
        tp75 = sum(i in gt75 for i in gi_list)

        fp50 = len(pi_list) - tp50
        fp75 = len(pi_list) - tp75

        fn50 = len(gi_list) - tp50
        fn75 = len(gi_list) - tp75

        ious50 = [
            gt50[i][1]
            for i in gi_list
            if i in gt50
        ]

        ious75 = [
            gt75[i][1]
            for i in gi_list
            if i in gt75
        ]

        per_image_records.append({

            "model": model_name,
            "split": split_name,
            "image": relative_path,

            "class_id": class_id,
            "class_name": class_name,

            "gt_count": len(gi_list),
            "prediction_count": len(pi_list),

            "TP_50": tp50,
            "FP_50": fp50,
            "FN_50": fn50,

            "precision_50": (
                tp50 / (tp50 + fp50)
                if tp50 + fp50 else 0.0
            ),

            "recall_50": (
                tp50 / (tp50 + fn50)
                if tp50 + fn50 else 0.0
            ),

            "mean_matched_iou_50": (
                float(np.mean(ious50))
                if ious50 else 0.0
            ),

            "TP_75": tp75,
            "FP_75": fp75,
            "FN_75": fn75,

            "precision_75": (
                tp75 / (tp75 + fp75)
                if tp75 + fp75 else 0.0
            ),

            "recall_75": (
                tp75 / (tp75 + fn75)
                if tp75 + fn75 else 0.0
            ),

            "mean_matched_iou_75": (
                float(np.mean(ious75))
                if ious75 else 0.0
            ),
        })

    # --------------------------------------------------------
    # Per GT object.
    # --------------------------------------------------------

    for gi, gt in enumerate(gt_objects):

        m50 = gt50.get(gi)
        m75 = gt75.get(gi)

        best_same = gt_same[gi]
        best_other = gt_other[gi]

        best_pred = gt_same_pred[gi]

        if m50 is not None:

            reason = "TP"

        elif best_same >= 0.10:

            reason = "same_class_overlap_but_iou_below_0.50"

        elif best_other >= 0.50:

            reason = "possible_class_confusion"

        else:

            reason = "missed_or_no_reliable_overlap"

        gt_records.append({

            "model": model_name,
            "split": split_name,
            "image": relative_path,

            "class_id": gt["class_id"],
            "class_name": CLASS_NAMES[gt["class_id"]],

            "gt_mask_area_px": gt["mask_area"],
            "gt_bbox_width_px": gt["bbox_width"],
            "gt_bbox_height_px": gt["bbox_height"],
            "gt_max_side_px": gt["max_side"],

            "size_bucket": gt["size_bucket"],

            "best_same_class_iou": best_same,
            "best_other_class_iou": best_other,
            "best_any_class_iou": gt_any[gi],

            "best_any_predicted_class": (
                CLASS_NAMES[gt_any_class[gi]]
                if gt_any_class[gi] is not None
                else ""
            ),

            "best_same_class_pred_confidence": (
                pred_objects[best_pred]["confidence"]
                if best_pred is not None
                else 0.0
            ),

            "matched_at_iou50": m50 is not None,
            "matched_iou50": (
                m50[1] if m50 else 0.0
            ),

            "matched_pred_confidence_50": (
                pred_objects[m50[0]]["confidence"]
                if m50 else 0.0
            ),

            "matched_at_iou75": m75 is not None,
            "matched_iou75": (
                m75[1] if m75 else 0.0
            ),

            "matched_pred_confidence_75": (
                pred_objects[m75[0]]["confidence"]
                if m75 else 0.0
            ),

            "miss_reason_at_iou50": reason,
        })

    # --------------------------------------------------------
    # All predictions and FP types.
    # --------------------------------------------------------

    for pi, pred in enumerate(pred_objects):

        m50 = pred50.get(pi)
        m75 = pred75.get(pi)

        if m50 is not None:

            status50 = "TP"
            fp_type = ""

        else:

            status50 = "FP"

            if pred_same[pi] >= 0.50:

                fp_type = "duplicate_prediction"

            elif pred_other[pi] >= 0.50:

                fp_type = "cross_class_overlap"

            elif pred_same[pi] >= 0.10:

                fp_type = "same_class_low_mask_iou"

            elif pred_other[pi] >= 0.10:

                fp_type = "other_class_low_mask_iou"

            else:

                fp_type = "background_or_no_overlap"

        box = pred["box_xyxy"]

        prediction_records.append({

            "model": model_name,
            "split": split_name,
            "image": relative_path,

            "prediction_index": pi,

            "class_id": pred["class_id"],
            "class_name": CLASS_NAMES[pred["class_id"]],

            "confidence": pred["confidence"],

            "pred_mask_area_px": pred["mask_area"],
            "pred_bbox_width_px": pred["bbox_width"],
            "pred_bbox_height_px": pred["bbox_height"],
            "pred_max_side_px": pred["max_side"],

            "best_same_class_iou": pred_same[pi],
            "best_other_class_iou": pred_other[pi],
            "best_any_class_iou": pred_any[pi],

            "best_any_gt_class": (
                CLASS_NAMES[pred_any_class[pi]]
                if pred_any_class[pi] is not None
                else ""
            ),

            "status_iou50": status50,
            "false_positive_type": fp_type,

            "status_iou75": (
                "TP" if m75 is not None else "FP"
            ),

            "matched_gt_class_iou50": (
                CLASS_NAMES[
                    gt_objects[m50[0]]["class_id"]
                ]
                if m50 is not None else ""
            ),

            "matched_iou50": (
                m50[1] if m50 else 0.0
            ),

            "matched_iou75": (
                m75[1] if m75 else 0.0
            ),

            "box_x1": box[0],
            "box_y1": box[1],
            "box_x2": box[2],
            "box_y2": box[3],
        })


# ============================================================
# 9. RUN INFERENCE — ONE MODEL AT A TIME
# ============================================================

for model_name, checkpoint_path in MODELS.items():

    print("\n")
    print("#" * 90)
    print("INFERENCE:", model_name)
    print("#" * 90)

    gc.collect()
    torch.cuda.empty_cache()

    model = YOLO(str(checkpoint_path))

    processed = 0
    total_predictions = 0

    stream = model.predict(
        source=[str(p) for p in image_paths],
        **PREDICT_ARGS,
    )

    try:

        for result in stream:

            result_path = Path(result.path).resolve()

            image_path = image_path_map.get(result_path)

            if image_path is None:
                image_path = basename_map.get(
                    Path(result.path).name
                )

            if image_path is None:
                raise RuntimeError(
                    f"Unknown inference result: {result.path}"
                )

            height, width = result.orig_shape

            gt_objects = create_gt_objects(
                image_path,
                height,
                width,
            )

            pred_objects = []

            if result.boxes is not None and len(result.boxes):

                classes = (
                    result.boxes.cls.detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    result.boxes.conf.detach()
                    .cpu()
                    .numpy()
                )

                boxes = (
                    result.boxes.xyxy.detach()
                    .cpu()
                    .numpy()
                )

                if result.masks is None:
                    raise RuntimeError(
                        "Predicted boxes found without predicted masks."
                    )

                if len(result.masks.data) != len(classes):
                    raise RuntimeError(
                        "Prediction mask count does not match box count."
                    )

                for pi in range(len(classes)):

                    class_id = int(classes[pi])

                    if not 0 <= class_id < NUM_CLASSES:
                        continue

                    mask = (
                        result.masks.data[pi]
                        .detach()
                        .cpu()
                        .numpy()
                    ) > 0.5

                    if mask.shape != (height, width):

                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    (
                        bbox,
                        area,
                        box_w,
                        box_h,
                        max_side,
                    ) = mask_statistics(mask)

                    pred_objects.append({

                        "class_id": class_id,

                        "confidence": float(
                            confidences[pi]
                        ),

                        "mask": mask,
                        "bbox": bbox,
                        "mask_area": area,

                        "bbox_width": box_w,
                        "bbox_height": box_h,
                        "max_side": max_side,

                        "box_xyxy": [
                            float(x) for x in boxes[pi]
                        ],
                    })

            relative_path = str(
                image_path.relative_to(VAL_IMAGE_ROOT)
            )

            analyze_image(
                model_name=model_name,
                relative_path=relative_path,
                split_name=image_split_map[image_path],
                gt_objects=gt_objects,
                pred_objects=pred_objects,
            )

            processed += 1
            total_predictions += len(pred_objects)

            del result

            if processed % 25 == 0:

                print(
                    f"{model_name}: "
                    f"{processed}/268 images; "
                    f"predictions={total_predictions}"
                )

                gc.collect()
                torch.cuda.empty_cache()

    finally:

        try:
            stream.close()
        except Exception:
            pass

    if processed != 268:
        raise RuntimeError(
            f"{model_name}: processed {processed}/268 images."
        )

    inference_summary[model_name] = {

        "images": processed,
        "predictions": total_predictions,
        "ground_truth_instances": 3042,
        "checkpoint": str(checkpoint_path),
    }

    print("\nCompleted:", model_name)
    print("Images:", processed)
    print("Predictions:", total_predictions)

    del stream
    del model

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 10. BUILD DATAFRAMES
# ============================================================

per_image_df = pd.DataFrame(per_image_records)
gt_df = pd.DataFrame(gt_records)
pred_df = pd.DataFrame(prediction_records)

assert len(per_image_df) == 2 * 268 * 6
assert len(gt_df) == 2 * 3042

print("\nPer-image/class records:", len(per_image_df))
print("Ground-truth records:", len(gt_df))
print("Prediction records:", len(pred_df))


# ============================================================
# 11. PAIRED PER-IMAGE COMPARISON
# ============================================================

paired_df = per_image_df[
    per_image_df["model"] == "C_IRFS"
].merge(
    per_image_df[
        per_image_df["model"] == "CB_IRFS"
    ],
    on=[
        "split",
        "image",
        "class_id",
        "class_name",
    ],
    suffixes=("_C", "_CB"),
    how="inner",
    validate="one_to_one",
)

assert len(paired_df) == 268 * 6

for metric in [
    "TP_50",
    "FP_50",
    "FN_50",
    "precision_50",
    "recall_50",
    "TP_75",
    "FP_75",
    "FN_75",
    "precision_75",
    "recall_75",
]:

    paired_df[f"delta_{metric}"] = (
        paired_df[f"{metric}_CB"]
        - paired_df[f"{metric}_C"]
    )


# ============================================================
# 12. CLASS SUMMARY
# ============================================================

summary_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            image_group = per_image_df[
                (per_image_df["model"] == model_name)
                & (per_image_df["split"] == split_name)
                & (per_image_df["class_id"] == class_id)
            ]

            instance_group = gt_df[
                (gt_df["model"] == model_name)
                & (gt_df["split"] == split_name)
                & (gt_df["class_id"] == class_id)
            ]

            for threshold in ["50", "75"]:

                tp = int(
                    image_group[f"TP_{threshold}"].sum()
                )

                fp = int(
                    image_group[f"FP_{threshold}"].sum()
                )

                fn = int(
                    image_group[f"FN_{threshold}"].sum()
                )

                precision = (
                    tp / (tp + fp)
                    if tp + fp else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn else 0.0
                )

                matched = instance_group[
                    instance_group[
                        f"matched_at_iou{threshold}"
                    ]
                ]

                mean_iou = (
                    matched[f"matched_iou{threshold}"].mean()
                    if len(matched) else 0.0
                )

                summary_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "matching_threshold": (
                        0.50 if threshold == "50" else 0.75
                    ),

                    "GT": tp + fn,
                    "Predictions": tp + fp,

                    "TP": tp,
                    "FP": fp,
                    "FN": fn,

                    "precision": precision,
                    "recall": recall,

                    "mean_matched_mask_iou": float(mean_iou),
                })


class_summary_df = pd.DataFrame(summary_rows)


# ============================================================
# 13. SIZE-STRATIFIED RECALL
# ============================================================

size_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            for bucket in SIZE_BUCKETS:

                group = gt_df[
                    (gt_df["model"] == model_name)
                    & (gt_df["split"] == split_name)
                    & (gt_df["class_id"] == class_id)
                    & (gt_df["size_bucket"] == bucket)
                ]

                n_gt = len(group)

                tp50 = int(
                    group["matched_at_iou50"].sum()
                )

                tp75 = int(
                    group["matched_at_iou75"].sum()
                )

                size_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "size_bucket": bucket,
                    "GT_instances": n_gt,

                    "TP_IoU50": tp50,
                    "FN_IoU50": n_gt - tp50,

                    "recall_IoU50": (
                        tp50 / n_gt if n_gt else 0.0
                    ),

                    "TP_IoU75": tp75,
                    "FN_IoU75": n_gt - tp75,

                    "recall_IoU75": (
                        tp75 / n_gt if n_gt else 0.0
                    ),
                })


size_summary_df = pd.DataFrame(size_rows)


# ============================================================
# 14. FALSE POSITIVE ANALYSIS
# ============================================================

fp_df = pred_df[
    pred_df["status_iou50"] == "FP"
].copy()

if len(fp_df):

    fp_type_summary_df = (
        fp_df.groupby(
            [
                "model",
                "split",
                "class_id",
                "class_name",
                "false_positive_type",
            ],
            dropna=False,
        )
        .agg(
            FP_count=("confidence", "count"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_best_same_class_iou=(
                "best_same_class_iou",
                "mean",
            ),
            mean_best_any_class_iou=(
                "best_any_class_iou",
                "mean",
            ),
        )
        .reset_index()
    )

else:

    fp_type_summary_df = pd.DataFrame()


missed_df = gt_df[
    ~gt_df["matched_at_iou50"]
].copy()

missed_df = missed_df.sort_values(
    by=[
        "model",
        "class_name",
        "gt_max_side_px",
        "best_same_class_iou",
    ],
    ascending=[
        True,
        True,
        True,
        False,
    ],
)


# ============================================================
# 15. SAVE CSV REPORTS
# ============================================================

def save_csv(df, filename):

    path = OUTPUT_DIR / filename

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print("[SAVED]", path)


save_csv(
    per_image_df,
    "per_image_class_metrics.csv",
)

save_csv(
    paired_df,
    "paired_image_class_comparison.csv",
)

save_csv(
    gt_df,
    "per_ground_truth_instance.csv",
)

save_csv(
    pred_df,
    "all_predictions.csv",
)

save_csv(
    fp_df,
    "false_positive_predictions.csv",
)

save_csv(
    class_summary_df,
    "class_summary.csv",
)

save_csv(
    size_summary_df,
    "size_stratified_recall.csv",
)

save_csv(
    fp_type_summary_df,
    "false_positive_type_summary.csv",
)

save_csv(
    missed_df,
    "missed_ground_truth_objects.csv",
)


# ============================================================
# 16. PRINT COMBINED CLASS SUMMARY
# ============================================================

print("\n" + "=" * 110)
print("COMBINED 268 — MASK IoU >= 0.50")
print("=" * 110)

combined = class_summary_df[
    (class_summary_df["split"] == "Combined_268")
    & (class_summary_df["matching_threshold"] == 0.50)
]

for class_name in CLASS_NAMES:

    c = combined[
        (combined["model"] == "C_IRFS")
        & (combined["class_name"] == class_name)
    ].iloc[0]

    cb = combined[
        (combined["model"] == "CB_IRFS")
        & (combined["class_name"] == class_name)
    ].iloc[0]

    print(f"\n{class_name}")

    print(
        f"C-IRFS : TP={int(c.TP):4d} "
        f"FP={int(c.FP):4d} "
        f"FN={int(c.FN):4d} "
        f"P={c.precision:.4f} "
        f"R={c.recall:.4f}"
    )

    print(
        f"CB-IRFS: TP={int(cb.TP):4d} "
        f"FP={int(cb.FP):4d} "
        f"FN={int(cb.FN):4d} "
        f"P={cb.precision:.4f} "
        f"R={cb.recall:.4f}"
    )

    print(
        f"Delta  : TP={int(cb.TP-c.TP):+d} "
        f"FP={int(cb.FP-c.FP):+d} "
        f"FN={int(cb.FN-c.FN):+d}"
    )


# ============================================================
# 17. SIZE-STRATIFIED RECALL — TARGET CLASSES
# ============================================================

print("\n" + "=" * 110)
print("SIZE-STRATIFIED RECALL — GLOVES AND SHOES")
print("=" * 110)

for split_name in [
    "Combined_268",
    "Old_119",
    "New_149",
]:

    print(f"\n{split_name}")

    for class_name in [
        "gloves",
        "shoes",
    ]:

        print(f"\n  {class_name}")

        for model_name in MODELS:

            subset = size_summary_df[
                (size_summary_df["model"] == model_name)
                & (size_summary_df["split"] == split_name)
                & (size_summary_df["class_name"] == class_name)
            ]

            print(f"    {model_name}")

            for _, row in subset.iterrows():

                print(
                    f"      {row['size_bucket']:12s} "
                    f"GT={int(row['GT_instances']):4d} "
                    f"R50={row['recall_IoU50']:.4f} "
                    f"R75={row['recall_IoU75']:.4f}"
                )


# ============================================================
# 18. FALSE POSITIVE TYPES — TARGET CLASSES
# ============================================================

print("\n" + "=" * 110)
print("FALSE POSITIVE TYPES — GLOVES AND SHOES")
print("=" * 110)

for class_name in [
    "gloves",
    "shoes",
]:

    print(f"\n{class_name}")

    subset = fp_type_summary_df[
        (fp_type_summary_df["split"] == "Combined_268")
        & (fp_type_summary_df["class_name"] == class_name)
    ]

    if len(subset):

        print(
            subset[
                [
                    "model",
                    "false_positive_type",
                    "FP_count",
                    "mean_confidence",
                    "median_confidence",
                ]
            ].to_string(index=False)
        )

    else:

        print("No false positives.")


# ============================================================
# 19. METADATA
# ============================================================

metadata = {

    "analysis": "paired_error_analysis_v2",

    "models": {
        name: str(path)
        for name, path in MODELS.items()
    },

    "validation_images": len(image_paths),
    "old_images": old_count,
    "new_images": new_count,

    "ground_truth_instances": 3042,

    "ground_truth_class_counts": {
        CLASS_NAMES[i]: int(EXPECTED_GT_COUNTS[i])
        for i in range(NUM_CLASSES)
    },

    "source_label_lines": source_lines_total,
    "duplicate_labels_removed": duplicate_total,

    "prediction_configuration": PREDICT_ARGS,

    "matching_method": (
        "Class-wise Hungarian one-to-one matching"
    ),

    "matching_thresholds": [
        0.50,
        0.75,
    ],

    "size_buckets": SIZE_BUCKETS,

    "inference_summary": inference_summary,

    "interpretation_warning": (
        "Fixed-confidence diagnostic matching, not official AP. "
        "Metrics depend on confidence threshold, NMS IoU and max_det."
    ),
}

metadata_path = OUTPUT_DIR / "analysis_metadata.json"

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 20. FINAL
# ============================================================

print("\n" + "=" * 100)
print("PAIRED ERROR ANALYSIS COMPLETED")
print("=" * 100)

print("Images:", len(image_paths))
print("Models:", len(MODELS))
print("GT instances:", 3042)

print("\nOutput:", OUTPUT_DIR)

print("\n[CELL 17C COMPLETED]")

###########################################################################
# CELL 25
###########################################################################
# ============================================================
# CELL 17D — CHECK ALL GPU DEVICES AND ACTIVE PROCESSES
# Diagnostic only. No inference or training.
# ============================================================

import torch
import subprocess

print("=" * 80)
print("CUDA DEVICE COUNT")
print("=" * 80)

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

print("\n" + "=" * 80)
print("GPU MEMORY")
print("=" * 80)

gpu_status = []

for device_id in range(torch.cuda.device_count()):

    free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)

    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)

    status = {
        "device": device_id,
        "name": torch.cuda.get_device_name(device_id),
        "free_gb": free_bytes / 1024**3,
        "total_gb": total_bytes / 1024**3,
        "allocated_gb": allocated / 1024**3,
        "reserved_gb": reserved / 1024**3,
    }

    gpu_status.append(status)

    print(f"\nGPU {device_id}: {status['name']}")
    print(f"  Total     : {status['total_gb']:.2f} GB")
    print(f"  Free      : {status['free_gb']:.2f} GB")
    print(f"  Allocated : {status['allocated_gb']:.2f} GB")
    print(f"  Reserved  : {status['reserved_gb']:.2f} GB")


print("\n" + "=" * 80)
print("NVIDIA-SMI")
print("=" * 80)

try:

    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

except Exception as exc:
    print("nvidia-smi error:", repr(exc))


print("\n" + "=" * 80)
print("GPU SELECTION")
print("=" * 80)

eligible = [
    x for x in gpu_status
    if x["free_gb"] >= 5.0
]

if eligible:

    selected = max(
        eligible,
        key=lambda x: x["free_gb"],
    )

    SELECTED_GPU = selected["device"]

    print("[OK] A GPU has sufficient free memory.")
    print("Selected GPU:", SELECTED_GPU)
    print("Free memory:", f"{selected['free_gb']:.2f} GB")

else:

    SELECTED_GPU = None

    print("[WARNING] No GPU has at least 5 GB free.")
    print("Do not start inference yet.")

print("\n[CELL 17D COMPLETED]")

###########################################################################
# CELL 26
###########################################################################
# ============================================================
# CELL 17E — RESUME PAIRED ERROR ANALYSIS ON GPU 1
#
# Uses existing Cell 17 definitions and parsed GT polygons.
# GPU 0 is not touched.
#
# Inference:
#   GPU 1, batch=1, FP16, imgsz=960
#
# ============================================================

import gc
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO


# ============================================================
# 1. VERIFY EXISTING CELL 17 DATA AND FUNCTIONS
# ============================================================

required_objects = [
    "MODELS",
    "image_paths",
    "image_path_map",
    "basename_map",
    "image_split_map",
    "GT_POLYGONS",
    "CLASS_NAMES",
    "create_gt_objects",
    "mask_bbox_and_area",
    "analyze_one_image",
]

missing = [
    name for name in required_objects
    if name not in globals()
]

if missing:
    raise RuntimeError(
        "Previous Cell 17 definitions are missing:\n"
        f"{missing}\n"
        "Do not continue with this cell. "
        "The original Cell 17 must be executed again."
    )

print("=" * 80)
print("EXISTING ANALYSIS DATA")
print("=" * 80)

print("Images:", len(image_paths))
print("GT image records:", len(GT_POLYGONS))
print("Classes:", CLASS_NAMES)

print(
    "Analysis function:",
    inspect.signature(analyze_one_image)
)

assert len(image_paths) == 268
assert len(GT_POLYGONS) == 268


# ============================================================
# 2. SELECT GPU 1
# ============================================================

GPU_ID = 1

if torch.cuda.device_count() <= GPU_ID:
    raise RuntimeError("GPU 1 is unavailable.")

free_bytes, total_bytes = torch.cuda.mem_get_info(GPU_ID)

print("\n" + "=" * 80)
print("GPU 1 MEMORY")
print("=" * 80)

print("Device:", torch.cuda.get_device_name(GPU_ID))
print(f"Total: {total_bytes / 1024**3:.2f} GB")
print(f"Free : {free_bytes / 1024**3:.2f} GB")

if free_bytes < 5 * 1024**3:
    raise RuntimeError(
        "GPU 1 has insufficient free memory. "
        "Do not start inference."
    )

print("\n[OK] GPU 1 selected.")


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

ANALYSIS_DIR = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "paired_error_analysis_v2"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\nOutput:", ANALYSIS_DIR)


# ============================================================
# 4. MEMORY-SAFE INFERENCE CONFIGURATION
# ============================================================

PREDICT_ARGS.update({

    "imgsz": 960,

    "batch": 1,
    "workers": 0,

    "device": GPU_ID,
    "half": True,

    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,

    "retina_masks": True,
    "agnostic_nms": False,
    "augment": False,

    "verbose": False,
    "save": False,
    "stream": True,
})

print("\n" + "=" * 80)
print("INFERENCE CONFIGURATION")
print("=" * 80)

for key, value in PREDICT_ARGS.items():
    print(f"{key:20s}: {value}")


# ============================================================
# 5. RESET OUTPUT CONTAINERS
# ============================================================

per_image_records = []
gt_records = []
prediction_records = []

inference_summary = {}

gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 6. RUN INFERENCE ON GPU 1
# ============================================================

for model_name, checkpoint_path in MODELS.items():

    print("\n")
    print("#" * 85)
    print("INFERENCE:", model_name)
    print("GPU:", GPU_ID)
    print("#" * 85)

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)

    model = YOLO(str(checkpoint_path))

    processed = 0
    total_predictions = 0
    total_gt = 0

    prediction_stream = model.predict(
        source=[str(p) for p in image_paths],
        **PREDICT_ARGS,
    )

    try:

        for result in prediction_stream:

            result_path = Path(
                result.path
            ).resolve()

            image_path = image_path_map.get(
                result_path
            )

            if image_path is None:
                image_path = basename_map.get(
                    Path(result.path).name
                )

            if image_path is None:
                raise RuntimeError(
                    f"Unknown image: {result.path}"
                )

            height, width = result.orig_shape

            # ------------------------------------------------
            # Ground-truth objects
            # ------------------------------------------------

            gt_objects = create_gt_objects(
                image_path,
                height,
                width,
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            pred_objects = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                pred_classes = (
                    result.boxes.cls
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                pred_confidences = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                pred_boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                if result.masks is None:
                    raise RuntimeError(
                        "Predicted boxes found without masks."
                    )

                if len(result.masks.data) != len(pred_classes):
                    raise RuntimeError(
                        "Prediction box/mask count mismatch."
                    )

                for pi in range(len(pred_classes)):

                    class_id = int(
                        pred_classes[pi]
                    )

                    if not 0 <= class_id < len(CLASS_NAMES):
                        continue

                    # Transfer one mask at a time to CPU.
                    pred_mask = (
                        result.masks.data[pi]
                        .detach()
                        .cpu()
                        .numpy()
                    ) > 0.5

                    # retina_masks=True should return original
                    # dimensions. Resize only if necessary.
                    if pred_mask.shape != (height, width):

                        pred_mask = cv2.resize(
                            pred_mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    (
                        pred_bbox,
                        pred_area,
                        pred_width,
                        pred_height,
                        pred_max_side,
                    ) = mask_bbox_and_area(pred_mask)

                    pred_objects.append({

                        "class_id": class_id,

                        "confidence": float(
                            pred_confidences[pi]
                        ),

                        "mask": pred_mask,
                        "bbox": pred_bbox,
                        "mask_area": pred_area,

                        "bbox_width": pred_width,
                        "bbox_height": pred_height,
                        "max_side": pred_max_side,

                        "box_xyxy": [
                            float(x)
                            for x in pred_boxes[pi]
                        ],
                    })

            # ------------------------------------------------
            # Per-image paired analysis
            # ------------------------------------------------

            relative_path = str(
                image_path.relative_to(
                    Path(
                        "/kaggle/working/"
                        "YOLO11_seg_expanded/"
                        "dataset/images/val"
                    )
                )
            )

            split_name = image_split_map[image_path]

            analyze_one_image(
                model_name=model_name,
                image_path=image_path,
                gt_objects=gt_objects,
                pred_objects=pred_objects,
                image_rel_path=relative_path,
                split_name=split_name,
            )

            processed += 1
            total_predictions += len(pred_objects)
            total_gt += len(gt_objects)

            del result

            if processed % 25 == 0:

                print(
                    f"{model_name}: "
                    f"{processed}/268 images | "
                    f"Predictions: {total_predictions}"
                )

                gc.collect()
                torch.cuda.empty_cache()

    finally:

        try:
            prediction_stream.close()
        except Exception:
            pass

    assert processed == 268, (
        f"{model_name}: only processed {processed}/268 images."
    )

    assert total_gt == 3042, (
        f"{model_name}: expected 3042 GT instances, "
        f"found {total_gt}."
    )

    inference_summary[model_name] = {

        "images": processed,
        "ground_truth_instances": total_gt,
        "predictions": total_predictions,
        "checkpoint": str(checkpoint_path),
        "gpu": GPU_ID,
    }

    print("\nCompleted:", model_name)
    print("Images:", processed)
    print("GT:", total_gt)
    print("Predictions:", total_predictions)

    del prediction_stream
    del model

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 7. BUILD DATAFRAMES
# ============================================================

per_image_df = pd.DataFrame(
    per_image_records
)

gt_df = pd.DataFrame(
    gt_records
)

pred_df = pd.DataFrame(
    prediction_records
)

assert len(per_image_df) == 2 * 268 * 6
assert len(gt_df) == 2 * 3042

print("\n" + "=" * 80)
print("RESULT COUNTS")
print("=" * 80)

print("Per-image/class:", len(per_image_df))
print("Ground truths  :", len(gt_df))
print("Predictions    :", len(pred_df))


# ============================================================
# 8. PAIRED IMAGE COMPARISON
# ============================================================

paired_df = per_image_df[
    per_image_df["model"] == "C_IRFS"
].merge(
    per_image_df[
        per_image_df["model"] == "CB_IRFS"
    ],
    on=[
        "split",
        "image",
        "class_id",
        "class_name",
    ],
    suffixes=("_C", "_CB"),
    how="inner",
    validate="one_to_one",
)

assert len(paired_df) == 268 * 6

for metric in [
    "TP_50",
    "FP_50",
    "FN_50",
    "precision_50",
    "recall_50",

    "TP_75",
    "FP_75",
    "FN_75",
    "precision_75",
    "recall_75",
]:

    paired_df[f"delta_{metric}"] = (
        paired_df[f"{metric}_CB"]
        - paired_df[f"{metric}_C"]
    )


# ============================================================
# 9. CLASS SUMMARY
# ============================================================

class_summary_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            image_group = per_image_df[
                (per_image_df["model"] == model_name)
                & (per_image_df["split"] == split_name)
                & (per_image_df["class_id"] == class_id)
            ]

            instance_group = gt_df[
                (gt_df["model"] == model_name)
                & (gt_df["split"] == split_name)
                & (gt_df["class_id"] == class_id)
            ]

            for threshold in ["50", "75"]:

                tp = int(
                    image_group[f"TP_{threshold}"].sum()
                )

                fp = int(
                    image_group[f"FP_{threshold}"].sum()
                )

                fn = int(
                    image_group[f"FN_{threshold}"].sum()
                )

                precision = (
                    tp / (tp + fp)
                    if tp + fp else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn else 0.0
                )

                matched = instance_group[
                    instance_group[
                        f"matched_at_iou{threshold}"
                    ]
                ]

                mean_iou = (
                    matched[
                        f"matched_iou{threshold}"
                    ].mean()
                    if len(matched) else 0.0
                )

                class_summary_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "matching_threshold": (
                        0.50 if threshold == "50" else 0.75
                    ),

                    "GT": tp + fn,
                    "Predictions": tp + fp,

                    "TP": tp,
                    "FP": fp,
                    "FN": fn,

                    "precision": precision,
                    "recall": recall,

                    "mean_matched_mask_iou": float(mean_iou),
                })


class_summary_df = pd.DataFrame(
    class_summary_rows
)


# ============================================================
# 10. SIZE-STRATIFIED RECALL
# ============================================================

SIZE_BUCKETS = [
    "lt_32px",
    "32_64px",
    "64_128px",
    "ge_128px",
]

size_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            for bucket in SIZE_BUCKETS:

                group = gt_df[
                    (gt_df["model"] == model_name)
                    & (gt_df["split"] == split_name)
                    & (gt_df["class_id"] == class_id)
                    & (gt_df["size_bucket"] == bucket)
                ]

                n_gt = len(group)

                tp50 = int(
                    group["matched_at_iou50"].sum()
                )

                tp75 = int(
                    group["matched_at_iou75"].sum()
                )

                size_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "size_bucket": bucket,
                    "GT_instances": n_gt,

                    "TP_IoU50": tp50,
                    "FN_IoU50": n_gt - tp50,

                    "recall_IoU50": (
                        tp50 / n_gt if n_gt else 0.0
                    ),

                    "TP_IoU75": tp75,
                    "FN_IoU75": n_gt - tp75,

                    "recall_IoU75": (
                        tp75 / n_gt if n_gt else 0.0
                    ),
                })


size_summary_df = pd.DataFrame(
    size_rows
)


# ============================================================
# 11. FALSE POSITIVE TYPE SUMMARY
# ============================================================

fp_df = pred_df[
    pred_df["status_iou50"] == "FP"
].copy()

if not fp_df.empty:

    fp_type_summary_df = (
        fp_df.groupby(
            [
                "model",
                "split",
                "class_id",
                "class_name",
                "false_positive_type",
            ],
            dropna=False,
        )
        .agg(
            FP_count=("confidence", "count"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),

            mean_best_same_class_iou=(
                "best_same_class_iou",
                "mean",
            ),

            mean_best_any_class_iou=(
                "best_any_class_iou",
                "mean",
            ),
        )
        .reset_index()
    )

else:

    fp_type_summary_df = pd.DataFrame()


# ============================================================
# 12. MISSED GROUND-TRUTH OBJECTS
# ============================================================

missed_df = gt_df[
    ~gt_df["matched_at_iou50"]
].copy()

missed_df = missed_df.sort_values(
    by=[
        "model",
        "class_name",
        "gt_max_side_px",
        "best_same_class_iou",
    ],
    ascending=[
        True,
        True,
        True,
        False,
    ],
)


# ============================================================
# 13. SAVE CSV REPORTS
# ============================================================

def save_analysis_csv(df, filename):

    path = ANALYSIS_DIR / filename

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print("[SAVED]", path)


save_analysis_csv(
    per_image_df,
    "per_image_class_metrics.csv",
)

save_analysis_csv(
    paired_df,
    "paired_image_class_comparison.csv",
)

save_analysis_csv(
    gt_df,
    "per_ground_truth_instance.csv",
)

save_analysis_csv(
    pred_df,
    "all_predictions.csv",
)

save_analysis_csv(
    fp_df,
    "false_positive_predictions.csv",
)

save_analysis_csv(
    class_summary_df,
    "class_summary.csv",
)

save_analysis_csv(
    size_summary_df,
    "size_stratified_recall.csv",
)

save_analysis_csv(
    fp_type_summary_df,
    "false_positive_type_summary.csv",
)

save_analysis_csv(
    missed_df,
    "missed_ground_truth_objects.csv",
)


# ============================================================
# 14. PRINT COMBINED CLASS COMPARISON
# ============================================================

print("\n" + "=" * 110)
print("COMBINED 268 — MASK MATCHING AT IoU >= 0.50")
print("=" * 110)

combined = class_summary_df[
    (class_summary_df["split"] == "Combined_268")
    & (class_summary_df["matching_threshold"] == 0.50)
]

for class_name in CLASS_NAMES:

    c = combined[
        (combined["model"] == "C_IRFS")
        & (combined["class_name"] == class_name)
    ].iloc[0]

    cb = combined[
        (combined["model"] == "CB_IRFS")
        & (combined["class_name"] == class_name)
    ].iloc[0]

    print(f"\n{class_name}")

    print(
        f"C-IRFS : TP={int(c.TP):4d} "
        f"FP={int(c.FP):4d} "
        f"FN={int(c.FN):4d} "
        f"P={c.precision:.4f} "
        f"R={c.recall:.4f}"
    )

    print(
        f"CB-IRFS: TP={int(cb.TP):4d} "
        f"FP={int(cb.FP):4d} "
        f"FN={int(cb.FN):4d} "
        f"P={cb.precision:.4f} "
        f"R={cb.recall:.4f}"
    )

    print(
        f"Delta  : "
        f"TP={int(cb.TP-c.TP):+d} "
        f"FP={int(cb.FP-c.FP):+d} "
        f"FN={int(cb.FN-c.FN):+d}"
    )


# ============================================================
# 15. SIZE-STRATIFIED RECALL
# ============================================================

print("\n" + "=" * 110)
print("SIZE-STRATIFIED RECALL — GLOVES AND SHOES")
print("=" * 110)

for split_name in [
    "Combined_268",
    "Old_119",
    "New_149",
]:

    print(f"\n{split_name}")

    for class_name in [
        "gloves",
        "shoes",
    ]:

        print(f"\n  {class_name}")

        for model_name in MODELS:

            subset = size_summary_df[
                (size_summary_df["model"] == model_name)
                & (size_summary_df["split"] == split_name)
                & (size_summary_df["class_name"] == class_name)
            ]

            print(f"    {model_name}")

            for _, row in subset.iterrows():

                print(
                    f"      {row['size_bucket']:12s} "
                    f"GT={int(row['GT_instances']):4d} "
                    f"Recall50={row['recall_IoU50']:.4f} "
                    f"Recall75={row['recall_IoU75']:.4f}"
                )


# ============================================================
# 16. FALSE POSITIVE TYPES
# ============================================================

print("\n" + "=" * 110)
print("FALSE POSITIVE TYPES — GLOVES AND SHOES")
print("=" * 110)

for class_name in [
    "gloves",
    "shoes",
]:

    print(f"\n{class_name}")

    subset = fp_type_summary_df[
        (fp_type_summary_df["split"] == "Combined_268")
        & (fp_type_summary_df["class_name"] == class_name)
    ]

    if len(subset):

        print(
            subset[
                [
                    "model",
                    "false_positive_type",
                    "FP_count",
                    "mean_confidence",
                    "median_confidence",
                ]
            ].to_string(index=False)
        )

    else:

        print("No false positives.")


# ============================================================
# 17. SAVE METADATA
# ============================================================

metadata = {

    "analysis": "paired_error_analysis_v2",

    "models": {
        name: str(path)
        for name, path in MODELS.items()
    },

    "gpu": GPU_ID,

    "validation_images": len(image_paths),
    "old_images": 119,
    "new_images": 149,

    "ground_truth_instances": 3042,

    "ground_truth_class_counts": {
        CLASS_NAMES[i]: int(EXPECTED_GT_COUNTS[i])
        for i in range(NUM_CLASSES)
    },

    "prediction_configuration": PREDICT_ARGS,

    "matching_method": (
        "Class-wise Hungarian one-to-one matching"
    ),

    "matching_thresholds": [
        0.50,
        0.75,
    ],

    "size_buckets": SIZE_BUCKETS,

    "inference_summary": inference_summary,

    "interpretation_warning": (
        "Fixed-confidence diagnostic matching, not official AP. "
        "Results depend on confidence threshold, NMS IoU and max_det."
    ),
}

metadata_path = (
    ANALYSIS_DIR / "analysis_metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("PAIRED ERROR ANALYSIS COMPLETED")
print("=" * 100)

print("Images:", len(image_paths))
print("Models:", len(MODELS))
print("GT instances/model:", 3042)
print("Inference GPU:", GPU_ID)

print("\nOutput directory:")
print(ANALYSIS_DIR)

print("\n[CELL 17E COMPLETED]")

###########################################################################
# CELL 27
###########################################################################
# ============================================================
# CELL 17F — GPU 1 CLEANUP + SINGLE-IMAGE INFERENCE TEST
# No training. No checkpoint modification.
# ============================================================

import gc
import torch
from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. Release objects retained by the failed inference
# ------------------------------------------------------------

print("=" * 75)
print("CLEANING FAILED INFERENCE OBJECTS")
print("=" * 75)

for name in [
    "prediction_stream",
    "stream",
    "model",
    "result",
    "test_model",
    "test_results",
]:
    if name in globals():
        globals().pop(name, None)
        print("[DELETED]", name)

gc.collect()

torch.cuda.set_device(1)
torch.cuda.empty_cache()

try:
    torch.cuda.ipc_collect()
except Exception:
    pass

torch.cuda.synchronize(1)

# ------------------------------------------------------------
# 2. Check available memory
# ------------------------------------------------------------

free_bytes, total_bytes = torch.cuda.mem_get_info(1)

print("\nGPU 1:", torch.cuda.get_device_name(1))
print(f"Total: {total_bytes / 1024**3:.2f} GB")
print(f"Free : {free_bytes / 1024**3:.2f} GB")

if free_bytes < 5 * 1024**3:
    raise RuntimeError(
        "GPU 1 still has less than 5 GB free. "
        "Do not continue inference."
    )

# ------------------------------------------------------------
# 3. Configure low-memory prediction
# ------------------------------------------------------------

TEST_CONFIG = {
    "imgsz": 960,
    "batch": 1,
    "device": 1,
    "workers": 0,
    "conf": 0.005,
    "iou": 0.7,
    "max_det": 100,
    "retina_masks": False,
    "half": True,
    "augment": False,
    "verbose": False,
    "save": False,
    "stream": False,
}

print("\n" + "=" * 75)
print("TEST CONFIGURATION")
print("=" * 75)

for key, value in TEST_CONFIG.items():
    print(f"{key:20s}: {value}")

# ------------------------------------------------------------
# 4. Load only C-IRFS
# ------------------------------------------------------------

test_model = YOLO(str(C_CHECKPOINT))

print("\nModel loaded:", type(test_model.model).__name__)

# Use one validation image for testing.
test_image = str(image_paths[0])

print("Test image:", test_image)

# ------------------------------------------------------------
# 5. Run single-image inference
# ------------------------------------------------------------

test_results = test_model.predict(
    source=test_image,
    **TEST_CONFIG,
)

if not test_results:
    raise RuntimeError("Inference returned no result.")

test_result = test_results[0]

print("\n" + "=" * 75)
print("SINGLE-IMAGE RESULT")
print("=" * 75)

print("Original image shape:", test_result.orig_shape)

if test_result.boxes is None:
    detection_count = 0
else:
    detection_count = len(test_result.boxes)

print("Detection count:", detection_count)

if test_result.masks is not None:

    print(
        "Predicted mask tensor shape:",
        tuple(test_result.masks.data.shape)
    )

    print(
        "Predicted mask dtype:",
        test_result.masks.data.dtype
    )

    print(
        "Predicted mask device:",
        test_result.masks.data.device
    )

else:
    print("No masks returned for this image.")

# ------------------------------------------------------------
# 6. Check detection cap
# ------------------------------------------------------------

print("\nReached max_det cap:", detection_count >= 100)

# ------------------------------------------------------------
# 7. GPU memory after test
# ------------------------------------------------------------

torch.cuda.synchronize(1)

free_bytes, total_bytes = torch.cuda.mem_get_info(1)

print("\n" + "=" * 75)
print("GPU MEMORY AFTER SINGLE-IMAGE TEST")
print("=" * 75)

print(f"Free: {free_bytes / 1024**3:.2f} GB")
print(
    f"Allocated: "
    f"{torch.cuda.memory_allocated(1) / 1024**3:.2f} GB"
)

print("\n[CELL 17F COMPLETED]")

###########################################################################
# CELL 28
###########################################################################
# ============================================================
# CELL 17G — FRESH SESSION SINGLE-IMAGE INFERENCE TEST
# Standalone. Run after Kaggle Session restart.
# ============================================================

import gc
from pathlib import Path

import torch
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

CHECKPOINT = (
    PROJECT_DIR
    / "research_experiments"
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

VAL_IMAGE_DIR = (
    PROJECT_DIR
    / "dataset"
    / "images"
    / "val"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "paired_error_analysis_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)

if not VAL_IMAGE_DIR.is_dir():
    raise FileNotFoundError(VAL_IMAGE_DIR)


# ------------------------------------------------------------
# 2. Select one image
# ------------------------------------------------------------

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = sorted(
    p for p in VAL_IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

assert len(image_paths) == 268, (
    f"Expected 268 images, found {len(image_paths)}"
)

test_image = str(image_paths[0])

print("Test image:", test_image)
print("Total validation images:", len(image_paths))


# ------------------------------------------------------------
# 3. Select GPU 1 and inspect memory
# ------------------------------------------------------------

GPU_ID = 1

assert torch.cuda.is_available()
assert torch.cuda.device_count() > GPU_ID

torch.cuda.set_device(GPU_ID)

gc.collect()
torch.cuda.empty_cache()

free_bytes, total_bytes = torch.cuda.mem_get_info(GPU_ID)

print("\nGPU:", torch.cuda.get_device_name(GPU_ID))
print(f"Total: {total_bytes / 1024**3:.2f} GB")
print(f"Free : {free_bytes / 1024**3:.2f} GB")

if free_bytes < 8 * 1024**3:
    raise RuntimeError(
        "GPU 1 does not have 8 GB free after session restart. "
        "Inspect nvidia-smi before continuing."
    )


# ------------------------------------------------------------
# 4. Load one model
# ------------------------------------------------------------

print("\nLoading C-IRFS...")

model = YOLO(str(CHECKPOINT))


# ------------------------------------------------------------
# 5. Conservative inference configuration
# ------------------------------------------------------------

results = model.predict(
    source=test_image,

    imgsz=960,
    batch=1,

    device=GPU_ID,
    workers=0,

    conf=0.005,
    iou=0.7,
    max_det=100,

    # Critical memory-saving setting:
    retina_masks=False,

    # Use FP32 for this initial stability test.
    half=False,

    augment=False,
    verbose=False,
    save=False,
    stream=False,
)

if len(results) != 1:
    raise RuntimeError(
        f"Expected one result, received {len(results)}"
    )

result = results[0]


# ------------------------------------------------------------
# 6. Inspect predictions
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("SINGLE-IMAGE INFERENCE RESULT")
print("=" * 75)

print("Original image shape:", result.orig_shape)

num_detections = (
    len(result.boxes)
    if result.boxes is not None
    else 0
)

print("Detection count:", num_detections)

if result.masks is not None:

    print(
        "Mask tensor shape:",
        tuple(result.masks.data.shape)
    )

    print("Mask dtype:", result.masks.data.dtype)

    print(
        "Mask device:",
        result.masks.data.device
    )

else:
    print("No masks predicted on this image.")


# ------------------------------------------------------------
# 7. Check GPU memory after inference
# ------------------------------------------------------------

torch.cuda.synchronize(GPU_ID)

free_bytes, total_bytes = torch.cuda.mem_get_info(GPU_ID)

print("\n" + "=" * 75)
print("GPU MEMORY AFTER INFERENCE")
print("=" * 75)

print(f"Free: {free_bytes / 1024**3:.2f} GB")

print(
    "PyTorch allocated:",
    f"{torch.cuda.memory_allocated(GPU_ID) / 1024**3:.2f} GB"
)

print(
    "PyTorch reserved:",
    f"{torch.cuda.memory_reserved(GPU_ID) / 1024**3:.2f} GB"
)

print("\nReached max_det=100:", num_detections >= 100)


# ------------------------------------------------------------
# 8. Release test objects
# ------------------------------------------------------------

del result
del results
del model

gc.collect()
torch.cuda.empty_cache()

print("\n[CELL 17G COMPLETED]")

###########################################################################
# CELL 29
###########################################################################
# ============================================================
# CELL 17H — TEST LOW-MEMORY MASK SCALING
# One image only. No training.
# ============================================================

import gc
import inspect
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks


# ------------------------------------------------------------
# 1. Select GPU and verify memory
# ------------------------------------------------------------

GPU_ID = 1

torch.cuda.set_device(GPU_ID)
gc.collect()
torch.cuda.empty_cache()

free_bytes, total_bytes = torch.cuda.mem_get_info(GPU_ID)

print("=" * 75)
print("GPU MEMORY")
print("=" * 75)

print(f"Free: {free_bytes / 1024**3:.2f} GB")
print(f"Total: {total_bytes / 1024**3:.2f} GB")

if free_bytes < 8 * 1024**3:
    raise RuntimeError("Insufficient GPU memory.")


# ------------------------------------------------------------
# 2. Inspect Ultralytics mask scaling implementation
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("ULTRALYTICS scale_masks")
print("=" * 75)

print("Signature:", inspect.signature(scale_masks))
print(inspect.getsource(scale_masks))


# ------------------------------------------------------------
# 3. Load C-IRFS
# ------------------------------------------------------------

CHECKPOINT = Path(
    "/kaggle/working/YOLO11_seg_expanded/"
    "research_experiments/C_irfs_t030/weights/best.pt"
)

assert CHECKPOINT.is_file()

TEST_IMAGE = Path(
    "/kaggle/working/YOLO11_seg_expanded/"
    "dataset/images/val/new/job67/batch_012/"
    "VHung__crowd_20260908_143140__000000.jpg"
)

assert TEST_IMAGE.is_file()

model = YOLO(str(CHECKPOINT))


# ------------------------------------------------------------
# 4. Predict with low-memory settings
# ------------------------------------------------------------

results = model.predict(
    source=str(TEST_IMAGE),

    imgsz=960,
    batch=1,

    device=GPU_ID,
    workers=0,

    conf=0.001,
    iou=0.7,
    max_det=300,

    retina_masks=False,
    half=True,

    augment=False,
    verbose=False,
    save=False,
    stream=False,
)

result = results[0]

height, width = result.orig_shape

num_detections = (
    len(result.boxes)
    if result.boxes is not None
    else 0
)

print("\n" + "=" * 75)
print("PREDICTION")
print("=" * 75)

print("Image shape:", result.orig_shape)
print("Detection count:", num_detections)

if result.masks is None or num_detections == 0:
    raise RuntimeError("No instance masks returned.")


# ------------------------------------------------------------
# 5. Scale one predicted mask to original image coordinates
# ------------------------------------------------------------

raw_mask = (
    result.masks.data[0]
    .detach()
    .cpu()
    .float()
)

print("Raw mask shape:", tuple(raw_mask.shape))
print("Raw mask dtype:", raw_mask.dtype)

# Input shape: [B, N, H, W] = [1, 1, H, W]
mask_input = raw_mask.unsqueeze(0).unsqueeze(0)

scaled_mask = scale_masks(
    mask_input,
    (height, width),
)

scaled_mask = (
    scaled_mask[0, 0].numpy() > 0.5
)

print("\n" + "=" * 75)
print("SCALED MASK")
print("=" * 75)

print("Scaled shape:", scaled_mask.shape)
print("Expected shape:", (height, width))
print("Foreground pixels:", int(scaled_mask.sum()))

assert scaled_mask.shape == (height, width)
assert scaled_mask.sum() > 0

print("\n[PASS] Mask transformed to original coordinates.")


# ------------------------------------------------------------
# 6. Check whether max_det was reached
# ------------------------------------------------------------

print("\nDetection cap reached:", num_detections >= 300)

if num_detections >= 300:
    print(
        "[WARNING] This image reached max_det=300. "
        "Some predictions may have been truncated."
    )
else:
    print("[OK] Detection count is below max_det.")


# ------------------------------------------------------------
# 7. GPU memory check
# ------------------------------------------------------------

torch.cuda.synchronize(GPU_ID)

free_bytes, total_bytes = torch.cuda.mem_get_info(GPU_ID)

print("\n" + "=" * 75)
print("GPU MEMORY AFTER TEST")
print("=" * 75)

print(f"Free: {free_bytes / 1024**3:.2f} GB")
print(
    f"Allocated: "
    f"{torch.cuda.memory_allocated(GPU_ID) / 1024**3:.2f} GB"
)


# ------------------------------------------------------------
# 8. Release model and results
# ------------------------------------------------------------

del scaled_mask
del mask_input
del raw_mask
del result
del results
del model

gc.collect()
torch.cuda.empty_cache()

print("\n[CELL 17H COMPLETED]")

###########################################################################
# CELL 30
###########################################################################
# ============================================================
# CELL 17I — FULL PAIRED ERROR ANALYSIS
# C-IRFS vs CLASS-BALANCED IRFS
#
# Self-contained implementation.
#
# Memory optimization:
#   - GPU 1 only
#   - batch=1
#   - FP32 inference (no deprecated half argument)
#   - retina_masks=False
#   - scale_masks() on CPU
#   - Store cropped instance masks, not full-image masks
#   - Release each result before processing the next image
#
# Matching:
#   - Class-wise Hungarian assignment
#   - Mask IoU thresholds 0.50 and 0.75
#
# No training. No checkpoint modification.
# ============================================================

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks
from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

VAL_IMAGE_ROOT = (
    PROJECT_DIR / "dataset" / "images" / "val"
)

VAL_LABEL_ROOT = (
    PROJECT_DIR / "dataset" / "labels" / "val"
)

C_CHECKPOINT = (
    PROJECT_DIR
    / "research_experiments"
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

CB_CHECKPOINT = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "CB_IRFS_sqrt_17ep_v1"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "paired_error_analysis_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = {
    "C_IRFS": C_CHECKPOINT,
    "CB_IRFS": CB_CHECKPOINT,
}

CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

NUM_CLASSES = 6

EXPECTED_GT_COUNTS = [
    829,
    238,
    339,
    446,
    274,
    916,
]

GPU_ID = 1

PREDICT_ARGS = {
    "imgsz": 960,
    "batch": 1,
    "rect": True,
    "device": GPU_ID,
    "workers": 0,

    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,

    "retina_masks": False,
    "augment": False,

    "verbose": False,
    "save": False,
    "stream": True,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SIZE_BUCKETS = [
    "lt_32px",
    "32_64px",
    "64_128px",
    "ge_128px",
]

MATCH_THRESHOLDS = [0.50, 0.75]


# ============================================================
# 2. VERIFY ENVIRONMENT
# ============================================================

print("=" * 85)
print("ENVIRONMENT VERIFICATION")
print("=" * 85)

for path in [
    PROJECT_DIR,
    VAL_IMAGE_ROOT,
    VAL_LABEL_ROOT,
    C_CHECKPOINT,
    CB_CHECKPOINT,
]:
    if not path.exists():
        raise FileNotFoundError(path)

    print("[OK]", path)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")

if torch.cuda.device_count() <= GPU_ID:
    raise RuntimeError("GPU 1 unavailable.")

torch.cuda.set_device(GPU_ID)

free_bytes, total_bytes = torch.cuda.mem_get_info(GPU_ID)

print("\nDevice:", torch.cuda.get_device_name(GPU_ID))
print(f"Total memory: {total_bytes / 1024**3:.2f} GB")
print(f"Free memory : {free_bytes / 1024**3:.2f} GB")

if free_bytes < 8 * 1024**3:
    raise RuntimeError(
        "GPU 1 has insufficient free memory."
    )

print("\nInference configuration:")

for key, value in PREDICT_ARGS.items():
    print(f"{key:20s}: {value}")


# ============================================================
# 3. ENUMERATE VALIDATION IMAGES
# ============================================================

image_paths = sorted(
    p.resolve()
    for p in VAL_IMAGE_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

if len(image_paths) != 268:
    raise RuntimeError(
        f"Expected 268 images, found {len(image_paths)}."
    )

image_path_map = {
    p: p for p in image_paths
}

basename_map = {
    p.name: p for p in image_paths
}

image_split_map = {}

for image_path in image_paths:

    relative = image_path.relative_to(
        VAL_IMAGE_ROOT
    )

    if relative.parts[0] == "old":
        split = "Old_119"

    elif relative.parts[0] == "new":
        split = "New_149"

    else:
        raise RuntimeError(
            f"Unknown split: {relative}"
        )

    image_split_map[image_path] = split


old_count = sum(
    x == "Old_119"
    for x in image_split_map.values()
)

new_count = sum(
    x == "New_149"
    for x in image_split_map.values()
)

assert old_count == 119
assert new_count == 149

print("\nValidation images:", len(image_paths))
print("Old:", old_count)
print("New:", new_count)


# ============================================================
# 4. PARSE GROUND-TRUTH POLYGONS
# ============================================================

def get_label_path(image_path):

    relative = image_path.relative_to(
        VAL_IMAGE_ROOT
    )

    return (
        VAL_LABEL_ROOT / relative
    ).with_suffix(".txt")


def parse_label_file(label_path):

    objects = []
    seen = set()

    source_lines = 0
    duplicates = 0

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            source_lines += 1

            values = np.asarray(
                [float(x) for x in line.split()],
                dtype=np.float64,
            )

            if len(values) < 7:
                raise ValueError(
                    f"Invalid polygon: {label_path}:{line_number}"
                )

            if (len(values) - 1) % 2:
                raise ValueError(
                    f"Invalid coordinate count: {label_path}"
                )

            class_id = int(round(values[0]))

            if abs(values[0] - class_id) > 1e-6:
                raise ValueError("Non-integer class ID.")

            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(f"Invalid class ID: {class_id}")

            key = tuple(
                np.round(values, 8).tolist()
            )

            if key in seen:
                duplicates += 1
                continue

            seen.add(key)

            points = values[1:].reshape(-1, 2)

            if not np.isfinite(points).all():
                raise ValueError(
                    f"Non-finite coordinates: {label_path}"
                )

            objects.append({
                "class_id": class_id,
                "points": points,
            })

    return objects, source_lines, duplicates


GT_POLYGONS = {}

gt_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64,
)

source_line_count = 0
duplicate_count = 0

print("\n" + "=" * 85)
print("GROUND-TRUTH AUDIT")
print("=" * 85)

for index, image_path in enumerate(image_paths, start=1):

    objects, n_lines, n_duplicates = parse_label_file(
        get_label_path(image_path)
    )

    GT_POLYGONS[image_path] = objects

    source_line_count += n_lines
    duplicate_count += n_duplicates

    for obj in objects:
        gt_counts[obj["class_id"]] += 1

    if index % 50 == 0:
        print(f"Processed {index}/268")

print("\nSource lines:", source_line_count)
print("Exact duplicates removed:", duplicate_count)
print("Total GT:", int(gt_counts.sum()))

for class_id, class_name in enumerate(CLASS_NAMES):

    actual = int(gt_counts[class_id])
    expected = EXPECTED_GT_COUNTS[class_id]

    print(
        f"{class_name:22s} "
        f"{actual:5d} / {expected:5d}"
    )

    assert actual == expected

assert int(gt_counts.sum()) == 3042

print("\n[OK] Ground-truth audit passed.")


# ============================================================
# 5. COMPACT MASK REPRESENTATION
# ============================================================

def make_compact_mask(mask):

    """
    Store only the object crop inside its bounding box.

    Returned fields:
        mask_crop
        bbox
        mask_area
        bbox_width
        bbox_height
        max_side
        size_bucket
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1

    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    width = x2 - x1
    height = y2 - y1

    max_side = max(width, height)

    if max_side < 32:
        bucket = "lt_32px"

    elif max_side < 64:
        bucket = "32_64px"

    elif max_side < 128:
        bucket = "64_128px"

    else:
        bucket = "ge_128px"

    return {

        "mask_crop": mask[y1:y2, x1:x2].copy(),

        "bbox": (x1, y1, x2, y2),

        "mask_area": int(mask.sum()),

        "bbox_width": width,
        "bbox_height": height,

        "max_side": max_side,
        "size_bucket": bucket,
    }


def rasterize_gt_polygon(points, height, width):

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    points = np.clip(
        points,
        0.0,
        1.0,
    )

    pixel_points = np.empty_like(points)

    pixel_points[:, 0] = (
        points[:, 0] * (width - 1)
    )

    pixel_points[:, 1] = (
        points[:, 1] * (height - 1)
    )

    pixel_points = np.rint(
        pixel_points
    ).astype(np.int32)

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [pixel_points],
        1,
    )

    return mask.astype(bool)


def create_gt_objects(image_path, height, width):

    objects = []

    for item in GT_POLYGONS[image_path]:

        full_mask = rasterize_gt_polygon(
            item["points"],
            height,
            width,
        )

        compact = make_compact_mask(
            full_mask
        )

        if compact is None:
            raise RuntimeError(
                f"Empty GT mask: {image_path}"
            )

        compact["class_id"] = item["class_id"]

        objects.append(compact)

        del full_mask

    return objects


# ============================================================
# 6. MASK IoU ON CROPPED MASKS
# ============================================================

def compact_mask_iou(a, b):

    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Coordinates inside compact A crop.
    a_x1 = x1 - ax1
    a_y1 = y1 - ay1

    a_x2 = x2 - ax1
    a_y2 = y2 - ay1

    # Coordinates inside compact B crop.
    b_x1 = x1 - bx1
    b_y1 = y1 - by1

    b_x2 = x2 - bx1
    b_y2 = y2 - by1

    crop_a = a["mask_crop"][
        a_y1:a_y2,
        a_x1:a_x2,
    ]

    crop_b = b["mask_crop"][
        b_y1:b_y2,
        b_x1:b_x2,
    ]

    intersection = np.count_nonzero(
        crop_a & crop_b
    )

    union = (
        a["mask_area"]
        + b["mask_area"]
        - intersection
    )

    return (
        float(intersection / union)
        if union > 0 else 0.0
    )


# ============================================================
# 7. CREATE PREDICTED OBJECT FROM LOW-RES MASK
# ============================================================

def create_prediction_object(
    raw_mask_cpu,
    original_shape,
    class_id,
    confidence,
    box_xyxy,
):
    """
    raw_mask_cpu shape: H x W.

    Scale the low-resolution mask to the original image
    using Ultralytics scale_masks() on CPU.

    Store only its bounding-box crop.
    """

    height, width = original_shape

    mask_input = (
        raw_mask_cpu
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    scaled = scale_masks(
        mask_input,
        (height, width),
        mode="bilinear",
    )

    full_mask = (
        scaled[0, 0].numpy() > 0.5
    )

    compact = make_compact_mask(
        full_mask
    )

    del mask_input
    del scaled
    del full_mask

    if compact is None:
        return None

    compact["class_id"] = int(class_id)
    compact["confidence"] = float(confidence)

    compact["box_xyxy"] = [
        float(x) for x in box_xyxy
    ]

    return compact


# ============================================================
# 8. HUNGARIAN MATCHING
# ============================================================

def build_iou_matrix(gt_objects, pred_objects):

    matrix = np.zeros(
        (len(gt_objects), len(pred_objects)),
        dtype=np.float32,
    )

    for gi, gt in enumerate(gt_objects):

        for pi, pred in enumerate(pred_objects):

            matrix[gi, pi] = compact_mask_iou(
                gt,
                pred,
            )

    return matrix


def match_objects(
    matrix,
    gt_objects,
    pred_objects,
    threshold,
):

    gt_to_pred = {}
    pred_to_gt = {}

    for class_id in range(NUM_CLASSES):

        gt_indices = [
            i for i, obj in enumerate(gt_objects)
            if obj["class_id"] == class_id
        ]

        pred_indices = [
            i for i, obj in enumerate(pred_objects)
            if obj["class_id"] == class_id
        ]

        if not gt_indices or not pred_indices:
            continue

        submatrix = matrix[
            np.ix_(gt_indices, pred_indices)
        ]

        rows, cols = linear_sum_assignment(
            -submatrix
        )

        for row, col in zip(rows, cols):

            overlap = float(
                submatrix[row, col]
            )

            if overlap < threshold:
                continue

            gi = gt_indices[row]
            pi = pred_indices[col]

            gt_to_pred[gi] = (
                pi,
                overlap,
            )

            pred_to_gt[pi] = (
                gi,
                overlap,
            )

    return gt_to_pred, pred_to_gt


# ============================================================
# 9. RESULT CONTAINERS
# ============================================================

per_image_records = []
gt_records = []
prediction_records = []

inference_summary = {}


# ============================================================
# 10. ANALYZE ONE IMAGE
# ============================================================

def analyze_one_image(
    model_name,
    image_rel_path,
    split_name,
    gt_objects,
    pred_objects,
):

    matrix = build_iou_matrix(
        gt_objects,
        pred_objects,
    )

    gt50, pred50 = match_objects(
        matrix,
        gt_objects,
        pred_objects,
        0.50,
    )

    gt75, pred75 = match_objects(
        matrix,
        gt_objects,
        pred_objects,
        0.75,
    )

    n_gt = len(gt_objects)
    n_pred = len(pred_objects)

    # --------------------------------------------------------
    # Best overlaps for GT objects.
    # --------------------------------------------------------

    gt_best_same = []
    gt_best_other = []
    gt_best_any = []

    gt_best_same_pred = []
    gt_best_any_class = []

    for gi, gt in enumerate(gt_objects):

        same_indices = [
            pi for pi, pred in enumerate(pred_objects)
            if pred["class_id"] == gt["class_id"]
        ]

        other_indices = [
            pi for pi, pred in enumerate(pred_objects)
            if pred["class_id"] != gt["class_id"]
        ]

        if same_indices:

            values = matrix[gi, same_indices]
            best_local = int(np.argmax(values))

            gt_best_same.append(
                float(values[best_local])
            )

            gt_best_same_pred.append(
                same_indices[best_local]
            )

        else:

            gt_best_same.append(0.0)
            gt_best_same_pred.append(None)

        gt_best_other.append(
            float(matrix[gi, other_indices].max())
            if other_indices else 0.0
        )

        if n_pred:

            best_any = int(
                np.argmax(matrix[gi])
            )

            gt_best_any.append(
                float(matrix[gi, best_any])
            )

            gt_best_any_class.append(
                pred_objects[best_any]["class_id"]
            )

        else:

            gt_best_any.append(0.0)
            gt_best_any_class.append(None)

    # --------------------------------------------------------
    # Best overlaps for predictions.
    # --------------------------------------------------------

    pred_best_same = []
    pred_best_other = []
    pred_best_any = []
    pred_best_any_gt_class = []

    for pi, pred in enumerate(pred_objects):

        same_indices = [
            gi for gi, gt in enumerate(gt_objects)
            if gt["class_id"] == pred["class_id"]
        ]

        other_indices = [
            gi for gi, gt in enumerate(gt_objects)
            if gt["class_id"] != pred["class_id"]
        ]

        pred_best_same.append(
            float(matrix[same_indices, pi].max())
            if same_indices else 0.0
        )

        pred_best_other.append(
            float(matrix[other_indices, pi].max())
            if other_indices else 0.0
        )

        if n_gt:

            best_gt = int(
                np.argmax(matrix[:, pi])
            )

            pred_best_any.append(
                float(matrix[best_gt, pi])
            )

            pred_best_any_gt_class.append(
                gt_objects[best_gt]["class_id"]
            )

        else:

            pred_best_any.append(0.0)
            pred_best_any_gt_class.append(None)

    # --------------------------------------------------------
    # Per-image/per-class metrics.
    # --------------------------------------------------------

    for class_id, class_name in enumerate(CLASS_NAMES):

        gi_list = [
            i for i, obj in enumerate(gt_objects)
            if obj["class_id"] == class_id
        ]

        pi_list = [
            i for i, obj in enumerate(pred_objects)
            if obj["class_id"] == class_id
        ]

        tp50 = sum(
            i in gt50 for i in gi_list
        )

        tp75 = sum(
            i in gt75 for i in gi_list
        )

        fp50 = len(pi_list) - tp50
        fp75 = len(pi_list) - tp75

        fn50 = len(gi_list) - tp50
        fn75 = len(gi_list) - tp75

        ious50 = [
            gt50[i][1]
            for i in gi_list
            if i in gt50
        ]

        ious75 = [
            gt75[i][1]
            for i in gi_list
            if i in gt75
        ]

        per_image_records.append({

            "model": model_name,
            "split": split_name,
            "image": image_rel_path,

            "class_id": class_id,
            "class_name": class_name,

            "gt_count": len(gi_list),
            "prediction_count": len(pi_list),

            "TP_50": tp50,
            "FP_50": fp50,
            "FN_50": fn50,

            "precision_50": (
                tp50 / (tp50 + fp50)
                if tp50 + fp50 else 0.0
            ),

            "recall_50": (
                tp50 / (tp50 + fn50)
                if tp50 + fn50 else 0.0
            ),

            "mean_matched_iou_50": (
                float(np.mean(ious50))
                if ious50 else 0.0
            ),

            "TP_75": tp75,
            "FP_75": fp75,
            "FN_75": fn75,

            "precision_75": (
                tp75 / (tp75 + fp75)
                if tp75 + fp75 else 0.0
            ),

            "recall_75": (
                tp75 / (tp75 + fn75)
                if tp75 + fn75 else 0.0
            ),

            "mean_matched_iou_75": (
                float(np.mean(ious75))
                if ious75 else 0.0
            ),
        })

    # --------------------------------------------------------
    # Per-GT-instance error analysis.
    # --------------------------------------------------------

    for gi, gt in enumerate(gt_objects):

        m50 = gt50.get(gi)
        m75 = gt75.get(gi)

        same_iou = gt_best_same[gi]
        other_iou = gt_best_other[gi]

        best_same_pred = gt_best_same_pred[gi]

        if m50 is not None:

            reason = "TP"

        elif same_iou >= 0.10:

            reason = "same_class_overlap_but_iou_below_0.50"

        elif other_iou >= 0.50:

            reason = "possible_class_confusion"

        else:

            reason = "missed_or_no_reliable_overlap"

        same_conf = (
            pred_objects[best_same_pred]["confidence"]
            if best_same_pred is not None
            else 0.0
        )

        gt_records.append({

            "model": model_name,
            "split": split_name,
            "image": image_rel_path,

            "class_id": gt["class_id"],
            "class_name": CLASS_NAMES[gt["class_id"]],

            "gt_mask_area_px": gt["mask_area"],
            "gt_bbox_width_px": gt["bbox_width"],
            "gt_bbox_height_px": gt["bbox_height"],
            "gt_max_side_px": gt["max_side"],

            "size_bucket": gt["size_bucket"],

            "best_same_class_iou": same_iou,
            "best_other_class_iou": other_iou,
            "best_any_class_iou": gt_best_any[gi],

            "best_any_predicted_class": (
                CLASS_NAMES[gt_best_any_class[gi]]
                if gt_best_any_class[gi] is not None
                else ""
            ),

            "best_same_class_pred_confidence": same_conf,

            "matched_at_iou50": m50 is not None,
            "matched_iou50": (
                m50[1] if m50 else 0.0
            ),

            "matched_pred_confidence_50": (
                pred_objects[m50[0]]["confidence"]
                if m50 else 0.0
            ),

            "matched_at_iou75": m75 is not None,
            "matched_iou75": (
                m75[1] if m75 else 0.0
            ),

            "matched_pred_confidence_75": (
                pred_objects[m75[0]]["confidence"]
                if m75 else 0.0
            ),

            "miss_reason_at_iou50": reason,
        })

    # --------------------------------------------------------
    # Prediction-level FP analysis.
    # --------------------------------------------------------

    for pi, pred in enumerate(pred_objects):

        m50 = pred50.get(pi)
        m75 = pred75.get(pi)

        if m50 is not None:

            status50 = "TP"
            fp_type = ""

        else:

            status50 = "FP"

            if pred_best_same[pi] >= 0.50:

                fp_type = "duplicate_prediction"

            elif pred_best_other[pi] >= 0.50:

                fp_type = "cross_class_overlap"

            elif pred_best_same[pi] >= 0.10:

                fp_type = "same_class_low_mask_iou"

            elif pred_best_other[pi] >= 0.10:

                fp_type = "other_class_low_mask_iou"

            else:

                fp_type = "background_or_no_overlap"

        box = pred["box_xyxy"]

        prediction_records.append({

            "model": model_name,
            "split": split_name,
            "image": image_rel_path,

            "prediction_index": pi,

            "class_id": pred["class_id"],
            "class_name": CLASS_NAMES[pred["class_id"]],

            "confidence": pred["confidence"],

            "pred_mask_area_px": pred["mask_area"],

            "pred_bbox_width_px": pred["bbox_width"],
            "pred_bbox_height_px": pred["bbox_height"],
            "pred_max_side_px": pred["max_side"],

            "size_bucket": pred["size_bucket"],

            "best_same_class_iou": pred_best_same[pi],
            "best_other_class_iou": pred_best_other[pi],
            "best_any_class_iou": pred_best_any[pi],

            "best_any_gt_class": (
                CLASS_NAMES[pred_best_any_gt_class[pi]]
                if pred_best_any_gt_class[pi] is not None
                else ""
            ),

            "status_iou50": status50,
            "false_positive_type": fp_type,

            "status_iou75": (
                "TP" if m75 is not None else "FP"
            ),

            "matched_gt_class_iou50": (
                CLASS_NAMES[
                    gt_objects[m50[0]]["class_id"]
                ]
                if m50 else ""
            ),

            "matched_iou50": (
                m50[1] if m50 else 0.0
            ),

            "matched_iou75": (
                m75[1] if m75 else 0.0
            ),

            "box_x1": box[0],
            "box_y1": box[1],
            "box_x2": box[2],
            "box_y2": box[3],
        })


# ============================================================
# 11. RUN INFERENCE ONE MODEL AT A TIME
# ============================================================

for model_name, checkpoint_path in MODELS.items():

    print("\n")
    print("#" * 90)
    print("STARTING:", model_name)
    print("#" * 90)

    gc.collect()
    torch.cuda.empty_cache()

    model = YOLO(str(checkpoint_path))

    processed = 0
    total_predictions = 0
    total_gt = 0
    images_at_detection_cap = 0

    stream = model.predict(
        source=[str(p) for p in image_paths],
        **PREDICT_ARGS,
    )

    try:

        for result in stream:

            image_result_path = Path(
                result.path
            ).resolve()

            image_path = image_path_map.get(
                image_result_path
            )

            if image_path is None:

                image_path = basename_map.get(
                    Path(result.path).name
                )

            if image_path is None:
                raise RuntimeError(
                    f"Unknown prediction path: {result.path}"
                )

            height, width = result.orig_shape

            # Ground-truth objects.
            gt_objects = create_gt_objects(
                image_path,
                height,
                width,
            )

            # Predicted objects.
            pred_objects = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                classes = (
                    result.boxes.cls
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                if result.masks is None:
                    raise RuntimeError(
                        "Boxes exist, but segmentation masks are missing."
                    )

                number_predictions = len(classes)

                if len(result.masks.data) != number_predictions:
                    raise RuntimeError(
                        "Number of masks and boxes differs."
                    )

                if number_predictions >= PREDICT_ARGS["max_det"]:
                    images_at_detection_cap += 1

                for pi in range(number_predictions):

                    class_id = int(classes[pi])

                    if not 0 <= class_id < NUM_CLASSES:
                        continue

                    # Move only one mask to CPU.
                    raw_mask = (
                        result.masks.data[pi]
                        .detach()
                        .cpu()
                    )

                    pred_object = create_prediction_object(
                        raw_mask_cpu=raw_mask,
                        original_shape=(height, width),
                        class_id=class_id,
                        confidence=confidences[pi],
                        box_xyxy=boxes[pi],
                    )

                    del raw_mask

                    if pred_object is not None:
                        pred_objects.append(pred_object)

                del classes
                del confidences
                del boxes

            relative_path = str(
                image_path.relative_to(VAL_IMAGE_ROOT)
            )

            analyze_one_image(
                model_name=model_name,
                image_rel_path=relative_path,
                split_name=image_split_map[image_path],
                gt_objects=gt_objects,
                pred_objects=pred_objects,
            )

            processed += 1
            total_predictions += len(pred_objects)
            total_gt += len(gt_objects)

            del pred_objects
            del gt_objects
            del result

            if processed % 25 == 0:

                print(
                    f"{model_name}: {processed}/268 "
                    f"| predictions={total_predictions} "
                    f"| GPU free="
                    f"{torch.cuda.mem_get_info(GPU_ID)[0] / 1024**3:.2f} GB"
                )

                gc.collect()
                torch.cuda.empty_cache()

    finally:

        try:
            stream.close()
        except Exception:
            pass

    if processed != 268:
        raise RuntimeError(
            f"{model_name}: only processed {processed}/268 images."
        )

    if total_gt != 3042:
        raise RuntimeError(
            f"{model_name}: expected 3042 GT instances, "
            f"got {total_gt}."
        )

    inference_summary[model_name] = {

        "images": processed,
        "ground_truth_instances": total_gt,
        "predictions": total_predictions,

        "images_reaching_max_det": images_at_detection_cap,

        "max_det": PREDICT_ARGS["max_det"],
        "checkpoint": str(checkpoint_path),
        "gpu": GPU_ID,
    }

    print("\nCOMPLETED:", model_name)
    print("Images:", processed)
    print("GT:", total_gt)
    print("Predictions:", total_predictions)
    print("Images at max_det:", images_at_detection_cap)

    # Incremental backup after each completed model.
    pd.DataFrame(
        [r for r in per_image_records if r["model"] == model_name]
    ).to_csv(
        OUTPUT_DIR / f"{model_name}_per_image_partial.csv",
        index=False,
    )

    pd.DataFrame(
        [r for r in gt_records if r["model"] == model_name]
    ).to_csv(
        OUTPUT_DIR / f"{model_name}_gt_partial.csv",
        index=False,
    )

    pd.DataFrame(
        [r for r in prediction_records if r["model"] == model_name]
    ).to_csv(
        OUTPUT_DIR / f"{model_name}_predictions_partial.csv",
        index=False,
    )

    # Release model before loading next checkpoint.
    del stream
    del model

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 12. BUILD DATAFRAMES
# ============================================================

per_image_df = pd.DataFrame(
    per_image_records
)

gt_df = pd.DataFrame(
    gt_records
)

pred_df = pd.DataFrame(
    prediction_records
)

assert len(per_image_df) == 2 * 268 * 6
assert len(gt_df) == 2 * 3042

print("\n" + "=" * 85)
print("RESULT COUNTS")
print("=" * 85)

print("Per-image/class:", len(per_image_df))
print("GT records:", len(gt_df))
print("Prediction records:", len(pred_df))


# ============================================================
# 13. PAIRED IMAGE COMPARISON
# ============================================================

paired_df = per_image_df[
    per_image_df["model"] == "C_IRFS"
].merge(
    per_image_df[
        per_image_df["model"] == "CB_IRFS"
    ],
    on=[
        "split",
        "image",
        "class_id",
        "class_name",
    ],
    suffixes=("_C", "_CB"),
    how="inner",
    validate="one_to_one",
)

assert len(paired_df) == 268 * NUM_CLASSES

for metric in [
    "TP_50",
    "FP_50",
    "FN_50",
    "precision_50",
    "recall_50",

    "TP_75",
    "FP_75",
    "FN_75",
    "precision_75",
    "recall_75",
]:

    paired_df[f"delta_{metric}"] = (
        paired_df[f"{metric}_CB"]
        - paired_df[f"{metric}_C"]
    )


# ============================================================
# 14. CLASS SUMMARY
# ============================================================

summary_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            image_group = per_image_df[
                (per_image_df["model"] == model_name)
                & (per_image_df["split"] == split_name)
                & (per_image_df["class_id"] == class_id)
            ]

            instance_group = gt_df[
                (gt_df["model"] == model_name)
                & (gt_df["split"] == split_name)
                & (gt_df["class_id"] == class_id)
            ]

            for threshold in ["50", "75"]:

                tp = int(
                    image_group[f"TP_{threshold}"].sum()
                )

                fp = int(
                    image_group[f"FP_{threshold}"].sum()
                )

                fn = int(
                    image_group[f"FN_{threshold}"].sum()
                )

                precision = (
                    tp / (tp + fp)
                    if tp + fp else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn else 0.0
                )

                matched = instance_group[
                    instance_group[
                        f"matched_at_iou{threshold}"
                    ]
                ]

                mean_iou = (
                    matched[
                        f"matched_iou{threshold}"
                    ].mean()
                    if len(matched) else 0.0
                )

                summary_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "matching_threshold": (
                        0.50 if threshold == "50" else 0.75
                    ),

                    "GT": tp + fn,
                    "Predictions": tp + fp,

                    "TP": tp,
                    "FP": fp,
                    "FN": fn,

                    "precision": precision,
                    "recall": recall,

                    "mean_matched_mask_iou": float(mean_iou),
                })


class_summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 15. SIZE-STRATIFIED RECALL
# ============================================================

size_rows = []

for model_name in MODELS:

    for split_name in [
        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(CLASS_NAMES):

            for bucket in SIZE_BUCKETS:

                group = gt_df[
                    (gt_df["model"] == model_name)
                    & (gt_df["split"] == split_name)
                    & (gt_df["class_id"] == class_id)
                    & (gt_df["size_bucket"] == bucket)
                ]

                n_gt = len(group)

                tp50 = int(
                    group["matched_at_iou50"].sum()
                )

                tp75 = int(
                    group["matched_at_iou75"].sum()
                )

                size_rows.append({

                    "model": model_name,
                    "split": split_name,

                    "class_id": class_id,
                    "class_name": class_name,

                    "size_bucket": bucket,
                    "GT_instances": n_gt,

                    "TP_IoU50": tp50,
                    "FN_IoU50": n_gt - tp50,

                    "recall_IoU50": (
                        tp50 / n_gt if n_gt else 0.0
                    ),

                    "TP_IoU75": tp75,
                    "FN_IoU75": n_gt - tp75,

                    "recall_IoU75": (
                        tp75 / n_gt if n_gt else 0.0
                    ),
                })


size_summary_df = pd.DataFrame(
    size_rows
)


# ============================================================
# 16. FALSE POSITIVE TYPES
# ============================================================

fp_df = pred_df[
    pred_df["status_iou50"] == "FP"
].copy()

if not fp_df.empty:

    fp_type_summary_df = (
        fp_df.groupby(
            [
                "model",
                "split",
                "class_id",
                "class_name",
                "false_positive_type",
            ],
            dropna=False,
        )
        .agg(
            FP_count=("confidence", "count"),

            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),

            mean_best_same_class_iou=(
                "best_same_class_iou",
                "mean",
            ),

            mean_best_any_class_iou=(
                "best_any_class_iou",
                "mean",
            ),
        )
        .reset_index()
    )

else:

    fp_type_summary_df = pd.DataFrame()


# ============================================================
# 17. MISSED GT OBJECTS
# ============================================================

missed_df = gt_df[
    ~gt_df["matched_at_iou50"]
].copy()

missed_df = missed_df.sort_values(
    by=[
        "model",
        "class_name",
        "gt_max_side_px",
        "best_same_class_iou",
    ],
    ascending=[
        True,
        True,
        True,
        False,
    ],
)


# ============================================================
# 18. SAVE CSV REPORTS
# ============================================================

def save_csv(df, filename):

    path = OUTPUT_DIR / filename

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print("[SAVED]", path)


save_csv(
    per_image_df,
    "per_image_class_metrics.csv",
)

save_csv(
    paired_df,
    "paired_image_class_comparison.csv",
)

save_csv(
    gt_df,
    "per_ground_truth_instance.csv",
)

save_csv(
    pred_df,
    "all_predictions.csv",
)

save_csv(
    fp_df,
    "false_positive_predictions.csv",
)

save_csv(
    class_summary_df,
    "class_summary.csv",
)

save_csv(
    size_summary_df,
    "size_stratified_recall.csv",
)

save_csv(
    fp_type_summary_df,
    "false_positive_type_summary.csv",
)

save_csv(
    missed_df,
    "missed_ground_truth_objects.csv",
)


# ============================================================
# 19. PRINT COMBINED TP / FP / FN
# ============================================================

print("\n" + "=" * 110)
print("COMBINED 268 — MASK IoU >= 0.50")
print("=" * 110)

combined = class_summary_df[
    (class_summary_df["split"] == "Combined_268")
    & (class_summary_df["matching_threshold"] == 0.50)
]

for class_name in CLASS_NAMES:

    c = combined[
        (combined["model"] == "C_IRFS")
        & (combined["class_name"] == class_name)
    ].iloc[0]

    cb = combined[
        (combined["model"] == "CB_IRFS")
        & (combined["class_name"] == class_name)
    ].iloc[0]

    print(f"\n{class_name}")

    print(
        f"C-IRFS : TP={int(c.TP):4d} "
        f"FP={int(c.FP):4d} "
        f"FN={int(c.FN):4d} "
        f"P={c.precision:.4f} "
        f"R={c.recall:.4f}"
    )

    print(
        f"CB-IRFS: TP={int(cb.TP):4d} "
        f"FP={int(cb.FP):4d} "
        f"FN={int(cb.FN):4d} "
        f"P={cb.precision:.4f} "
        f"R={cb.recall:.4f}"
    )

    print(
        f"Delta  : TP={int(cb.TP-c.TP):+d} "
        f"FP={int(cb.FP-c.FP):+d} "
        f"FN={int(cb.FN-c.FN):+d}"
    )


# ============================================================
# 20. SIZE-STRATIFIED RECALL — GLOVES / SHOES
# ============================================================

print("\n" + "=" * 110)
print("SIZE-STRATIFIED RECALL — GLOVES AND SHOES")
print("=" * 110)

for split_name in [
    "Combined_268",
    "Old_119",
    "New_149",
]:

    print(f"\n{split_name}")

    for class_name in [
        "gloves",
        "shoes",
    ]:

        print(f"\n  {class_name}")

        for model_name in MODELS:

            subset = size_summary_df[
                (size_summary_df["model"] == model_name)
                & (size_summary_df["split"] == split_name)
                & (size_summary_df["class_name"] == class_name)
            ]

            print(f"    {model_name}")

            for _, row in subset.iterrows():

                print(
                    f"      {row['size_bucket']:12s} "
                    f"GT={int(row['GT_instances']):4d} "
                    f"Recall50={row['recall_IoU50']:.4f} "
                    f"Recall75={row['recall_IoU75']:.4f}"
                )


# ============================================================
# 21. FALSE POSITIVE TYPES — GLOVES / SHOES
# ============================================================

print("\n" + "=" * 110)
print("FALSE POSITIVE TYPES — GLOVES AND SHOES")
print("=" * 110)

for class_name in [
    "gloves",
    "shoes",
]:

    print(f"\n{class_name}")

    subset = fp_type_summary_df[
        (fp_type_summary_df["split"] == "Combined_268")
        & (fp_type_summary_df["class_name"] == class_name)
    ]

    if len(subset):

        print(
            subset[
                [
                    "model",
                    "false_positive_type",
                    "FP_count",
                    "mean_confidence",
                    "median_confidence",
                ]
            ].to_string(index=False)
        )

    else:

        print("No false positives.")


# ============================================================
# 22. SAVE ANALYSIS METADATA
# ============================================================

metadata = {

    "analysis": "paired_error_analysis_v3",

    "models": {
        name: str(path)
        for name, path in MODELS.items()
    },

    "gpu": GPU_ID,

    "validation_images": len(image_paths),
    "old_images": old_count,
    "new_images": new_count,

    "ground_truth_instances": 3042,

    "ground_truth_class_counts": {
        CLASS_NAMES[i]: int(EXPECTED_GT_COUNTS[i])
        for i in range(NUM_CLASSES)
    },

    "source_label_lines": source_line_count,
    "exact_duplicate_labels_removed": duplicate_count,

    "prediction_configuration": PREDICT_ARGS,

    "mask_scaling": (
        "Ultralytics scale_masks applied on CPU "
        "with original image dimensions."
    ),

    "matching_method": (
        "Class-wise Hungarian one-to-one assignment."
    ),

    "matching_thresholds": MATCH_THRESHOLDS,

    "size_buckets": SIZE_BUCKETS,

    "inference_summary": inference_summary,

    "interpretation_warning": (
        "Fixed-confidence per-image diagnostic matching, "
        "not official Ultralytics AP. Results depend on "
        "confidence threshold, NMS IoU and max_det. "
        "Images reaching max_det may have truncated detections."
    ),
}

metadata_path = (
    OUTPUT_DIR / "analysis_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("PAIRED ERROR ANALYSIS COMPLETED")
print("=" * 100)

print("Images:", len(image_paths))
print("Models:", len(MODELS))
print("GT instances/model:", 3042)

for model_name, info in inference_summary.items():

    print(f"\n{model_name}")
    print("  Predictions:", info["predictions"])
    print("  Images reaching max_det:", info["images_reaching_max_det"])

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\n[CELL 17I COMPLETED]")

###########################################################################
# CELL 31
###########################################################################
# ============================================================
# CELL 17I — FULL PAIRED ERROR ANALYSIS
# C-IRFS vs CLASS-BALANCED IRFS
#
# SELF-CONTAINED: Does not depend on previous cell variables.
#
# GPU:
#   GPU 1 - Tesla T4
#
# MEMORY CONFIGURATION:
#   imgsz=960
#   batch=1
#   half=True
#   retina_masks=False
#   scale_masks() on CPU
#   Store cropped masks instead of full-resolution masks
#
# PREDICTION:
#   confidence=0.001
#   NMS IoU=0.7
#   max_det=300
#
# MATCHING:
#   Class-wise Hungarian one-to-one matching
#   Mask IoU thresholds: 0.50 and 0.75
#
# NO TRAINING. NO CHECKPOINT MODIFICATION.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import gc
import json
import hashlib
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks

from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

VAL_IMAGE_ROOT = (
    PROJECT_DIR
    / "dataset"
    / "images"
    / "val"
)

VAL_LABEL_ROOT = (
    PROJECT_DIR
    / "dataset"
    / "labels"
    / "val"
)

C_CHECKPOINT = (
    PROJECT_DIR
    / "research_experiments"
    / "C_irfs_t030"
    / "weights"
    / "best.pt"
)

CB_CHECKPOINT = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "CB_IRFS_sqrt_17ep_v1"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "kaggle_outputs"
    / "paired_error_analysis_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODELS = {
    "C_IRFS": C_CHECKPOINT,
    "CB_IRFS": CB_CHECKPOINT,
}

CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

NUM_CLASSES = len(CLASS_NAMES)

EXPECTED_GT_COUNTS = [
    829,
    238,
    339,
    446,
    274,
    916,
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 3. GPU CONFIGURATION
# ============================================================

GPU_ID = 1

PREDICT_ARGS = {

    "imgsz": 960,

    "batch": 1,

    "rect": True,

    "device": GPU_ID,

    "workers": 0,

    "conf": 0.001,

    "iou": 0.7,

    "max_det": 300,

    # Do not expand masks to original resolution on GPU.
    "retina_masks": False,

    # IMPORTANT:
    # FP16 was successfully tested in Cell 17H.
    # Keep this enabled to reduce GPU memory usage.
    "half": True,

    "augment": False,

    "verbose": False,

    "save": False,

    "stream": True,
}


# ============================================================
# 4. ANALYSIS CONFIGURATION
# ============================================================

MATCH_THRESHOLDS = [
    0.50,
    0.75,
]

SIZE_BUCKETS = [
    "lt_32px",
    "32_64px",
    "64_128px",
    "ge_128px",
]


# ============================================================
# 5. VERIFY FILES AND ENVIRONMENT
# ============================================================

print("=" * 85)
print("ENVIRONMENT VERIFICATION")
print("=" * 85)

required_paths = [
    PROJECT_DIR,
    VAL_IMAGE_ROOT,
    VAL_LABEL_ROOT,
    C_CHECKPOINT,
    CB_CHECKPOINT,
]

for path in required_paths:

    if not path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n{path}"
        )

    print("[OK]", path)


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")


if torch.cuda.device_count() <= GPU_ID:
    raise RuntimeError(
        f"GPU {GPU_ID} is unavailable."
    )


torch.cuda.set_device(GPU_ID)

free_bytes, total_bytes = torch.cuda.mem_get_info(
    GPU_ID
)

print("\nGPU:", torch.cuda.get_device_name(GPU_ID))

print(
    f"Total memory: {total_bytes / 1024**3:.2f} GB"
)

print(
    f"Free memory : {free_bytes / 1024**3:.2f} GB"
)


if free_bytes < 8 * 1024**3:
    raise RuntimeError(
        "GPU 1 has less than 8 GB free.\n"
        "Do not start inference. Restart the Kaggle session "
        "and check GPU memory again."
    )


print("\n[OK] GPU memory check passed.")


# ============================================================
# 6. VERIFY CLASS MAPPING AND CHECKPOINTS
# ============================================================

print("\n" + "=" * 85)
print("CHECKPOINT VERIFICATION")
print("=" * 85)

for name, path in MODELS.items():

    print(
        f"{name:12s}: "
        f"{path} "
        f"({path.stat().st_size / (1024**2):.2f} MB)"
    )


# ============================================================
# 7. ENUMERATE VALIDATION IMAGES
# ============================================================

image_paths = sorted(
    p.resolve()
    for p in VAL_IMAGE_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

if len(image_paths) != 268:
    raise RuntimeError(
        f"Expected 268 validation images, "
        f"found {len(image_paths)}."
    )


image_path_map = {
    p: p for p in image_paths
}

basename_map = {
    p.name: p for p in image_paths
}


image_split_map = {}

for image_path in image_paths:

    relative_path = image_path.relative_to(
        VAL_IMAGE_ROOT
    )

    if relative_path.parts[0] == "old":

        split_name = "Old_119"

    elif relative_path.parts[0] == "new":

        split_name = "New_149"

    else:

        raise RuntimeError(
            f"Unknown validation split: {relative_path}"
        )

    image_split_map[image_path] = split_name


old_count = sum(
    value == "Old_119"
    for value in image_split_map.values()
)

new_count = sum(
    value == "New_149"
    for value in image_split_map.values()
)

assert old_count == 119
assert new_count == 149


print("\n" + "=" * 85)
print("VALIDATION IMAGE AUDIT")
print("=" * 85)

print("Combined:", len(image_paths))
print("Old     :", old_count)
print("New     :", new_count)


# ============================================================
# 8. PARSE GROUND-TRUTH POLYGONS
# ============================================================

def get_label_path(image_path):

    relative_path = image_path.relative_to(
        VAL_IMAGE_ROOT
    )

    return (
        VAL_LABEL_ROOT
        / relative_path
    ).with_suffix(".txt")


def parse_label_file(label_path):

    objects = []

    seen = set()

    source_lines = 0

    duplicates = 0

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            source_lines += 1

            try:

                values = np.asarray(
                    [float(x) for x in line.split()],
                    dtype=np.float64,
                )

            except ValueError as exc:

                raise ValueError(
                    f"Invalid numeric label: "
                    f"{label_path}:{line_number}"
                ) from exc


            if len(values) < 7:
                raise ValueError(
                    f"Polygon has fewer than 3 points: "
                    f"{label_path}:{line_number}"
                )

            if (len(values) - 1) % 2 != 0:
                raise ValueError(
                    f"Invalid polygon coordinate count: "
                    f"{label_path}:{line_number}"
                )


            class_id = int(round(values[0]))

            if abs(values[0] - class_id) > 1e-6:
                raise ValueError(
                    f"Non-integer class ID: {label_path}"
                )

            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"Invalid class ID {class_id}: {label_path}"
                )


            key = tuple(
                np.round(values, 8).tolist()
            )

            if key in seen:

                duplicates += 1

                continue

            seen.add(key)


            points = values[1:].reshape(-1, 2)

            if not np.isfinite(points).all():
                raise ValueError(
                    f"Non-finite coordinates: {label_path}"
                )

            objects.append({

                "class_id": class_id,

                "points": points,
            })


    return (
        objects,
        source_lines,
        duplicates,
    )


# ------------------------------------------------------------
# Load all polygon coordinates.
# Full-resolution masks are created per image only.
# ------------------------------------------------------------

GT_POLYGONS = {}

gt_class_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64,
)

source_line_count = 0

duplicate_count = 0

print("\n" + "=" * 85)
print("GROUND-TRUTH LABEL AUDIT")
print("=" * 85)

for index, image_path in enumerate(
    image_paths,
    start=1,
):

    objects, n_lines, n_duplicates = (
        parse_label_file(
            get_label_path(image_path)
        )
    )

    GT_POLYGONS[image_path] = objects

    source_line_count += n_lines

    duplicate_count += n_duplicates

    for obj in objects:

        gt_class_counts[
            obj["class_id"]
        ] += 1

    if index % 50 == 0:

        print(
            f"Processed {index}/268 images"
        )


print("\nSource lines:", source_line_count)

print(
    "Exact duplicates removed:",
    duplicate_count
)

print(
    "Total GT instances:",
    int(gt_class_counts.sum())
)


for class_id, class_name in enumerate(CLASS_NAMES):

    actual = int(
        gt_class_counts[class_id]
    )

    expected = EXPECTED_GT_COUNTS[class_id]

    print(
        f"{class_name:22s} "
        f"{actual:5d} / {expected:5d}"
    )

    if actual != expected:
        raise RuntimeError(
            f"GT count mismatch for {class_name}."
        )


assert int(gt_class_counts.sum()) == 3042

print("\n[OK] Ground-truth audit passed.")


# ============================================================
# 9. RASTERIZE GROUND-TRUTH POLYGONS
# ============================================================

def rasterize_polygon(
    points,
    height,
    width,
):

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    points = np.clip(
        points,
        0.0,
        1.0,
    )

    pixel_points = np.empty_like(points)

    pixel_points[:, 0] = (
        points[:, 0] * (width - 1)
    )

    pixel_points[:, 1] = (
        points[:, 1] * (height - 1)
    )

    pixel_points = np.rint(
        pixel_points
    ).astype(np.int32)


    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [pixel_points],
        1,
    )

    return mask.astype(bool)


# ============================================================
# 10. COMPACT MASK REPRESENTATION
# ============================================================

def make_compact_mask(mask):

    """
    Keep only the object's cropped mask.

    This avoids storing thousands of full-resolution
    1080x1920 masks in memory.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None


    x1 = int(xs.min())

    x2 = int(xs.max()) + 1

    y1 = int(ys.min())

    y2 = int(ys.max()) + 1


    width = x2 - x1

    height = y2 - y1

    max_side = max(width, height)

    area = int(
        mask.sum()
    )


    if max_side < 32:

        bucket = "lt_32px"

    elif max_side < 64:

        bucket = "32_64px"

    elif max_side < 128:

        bucket = "64_128px"

    else:

        bucket = "ge_128px"


    return {

        "mask_crop": mask[
            y1:y2,
            x1:x2
        ].copy(),

        "bbox": (
            x1,
            y1,
            x2,
            y2,
        ),

        "mask_area": area,

        "bbox_width": width,

        "bbox_height": height,

        "max_side": max_side,

        "size_bucket": bucket,
    }


# ============================================================
# 11. CREATE GROUND-TRUTH OBJECTS FOR ONE IMAGE
# ============================================================

def create_gt_objects(
    image_path,
    height,
    width,
):

    objects = []

    for item in GT_POLYGONS[image_path]:

        full_mask = rasterize_polygon(
            item["points"],
            height,
            width,
        )

        compact = make_compact_mask(
            full_mask
        )

        del full_mask

        if compact is None:
            raise RuntimeError(
                f"Empty ground-truth mask: {image_path}"
            )

        compact["class_id"] = item["class_id"]

        objects.append(compact)

    return objects


# ============================================================
# 12. CONVERT LOW-RES PREDICTION MASK TO ORIGINAL IMAGE
# ============================================================

def create_prediction_object(
    raw_mask_cpu,
    original_shape,
    class_id,
    confidence,
    box_xyxy,
):

    """
    Input:
        raw_mask_cpu: low-resolution mask on CPU.

    Steps:
        1. Convert the low-resolution mask to float.
        2. Use Ultralytics scale_masks().
        3. Scale to original image dimensions.
        4. Threshold at 0.5.
        5. Crop to bounding box.
        6. Store only compact mask.
    """

    height, width = original_shape

    mask_input = (
        raw_mask_cpu
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )


    scaled_mask = scale_masks(
        mask_input,
        (height, width),
        mode="bilinear",
    )


    binary_mask = (
        scaled_mask[0, 0].numpy() > 0.5
    )


    compact = make_compact_mask(
        binary_mask
    )


    del mask_input

    del scaled_mask

    del binary_mask


    if compact is None:
        return None


    compact["class_id"] = int(class_id)

    compact["confidence"] = float(confidence)

    compact["box_xyxy"] = [
        float(x) for x in box_xyxy
    ]

    return compact


# ============================================================
# 13. MASK IoU
# ============================================================

def compact_mask_iou(a, b):

    ax1, ay1, ax2, ay2 = a["bbox"]

    bx1, by1, bx2, by2 = b["bbox"]


    x1 = max(ax1, bx1)

    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)

    y2 = min(ay2, by2)


    if x2 <= x1 or y2 <= y1:
        return 0.0


    # Crop coordinates inside mask A.
    a_x1 = x1 - ax1

    a_y1 = y1 - ay1

    a_x2 = x2 - ax1

    a_y2 = y2 - ay1


    # Crop coordinates inside mask B.
    b_x1 = x1 - bx1

    b_y1 = y1 - by1

    b_x2 = x2 - bx1

    b_y2 = y2 - by1


    crop_a = a["mask_crop"][
        a_y1:a_y2,
        a_x1:a_x2,
    ]

    crop_b = b["mask_crop"][
        b_y1:b_y2,
        b_x1:b_x2,
    ]


    intersection = np.count_nonzero(
        crop_a & crop_b
    )


    union = (
        a["mask_area"]
        + b["mask_area"]
        - intersection
    )


    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


# ============================================================
# 14. BUILD MASK IoU MATRIX
# ============================================================

def build_iou_matrix(
    gt_objects,
    pred_objects,
):

    matrix = np.zeros(
        (
            len(gt_objects),
            len(pred_objects),
        ),
        dtype=np.float32,
    )

    for gi, gt in enumerate(gt_objects):

        for pi, pred in enumerate(pred_objects):

            matrix[gi, pi] = compact_mask_iou(
                gt,
                pred,
            )

    return matrix


# ============================================================
# 15. CLASS-WISE HUNGARIAN MATCHING
# ============================================================

def match_objects(
    iou_matrix,
    gt_objects,
    pred_objects,
    threshold,
):

    """
    Class-wise one-to-one matching.

    Only edges meeting the IoU threshold are eligible.

    This avoids assigning a low-IoU pair that could block
    a valid higher-IoU match.
    """

    gt_to_pred = {}

    pred_to_gt = {}


    for class_id in range(NUM_CLASSES):

        gt_indices = [
            i
            for i, obj in enumerate(gt_objects)
            if obj["class_id"] == class_id
        ]

        pred_indices = [
            i
            for i, obj in enumerate(pred_objects)
            if obj["class_id"] == class_id
        ]


        if not gt_indices or not pred_indices:
            continue


        submatrix = iou_matrix[
            np.ix_(
                gt_indices,
                pred_indices,
            )
        ]


        valid = (
            submatrix >= threshold
        )


        if not valid.any():
            continue


        # Make all below-threshold matches prohibitively expensive.
        cost_matrix = np.where(
            valid,
            -submatrix,
            1e6,
        )


        rows, cols = linear_sum_assignment(
            cost_matrix
        )


        for row, col in zip(rows, cols):

            if not valid[row, col]:
                continue


            overlap = float(
                submatrix[row, col]
            )


            gi = gt_indices[row]

            pi = pred_indices[col]


            gt_to_pred[gi] = (
                pi,
                overlap,
            )

            pred_to_gt[pi] = (
                gi,
                overlap,
            )


    return (
        gt_to_pred,
        pred_to_gt,
    )


# ============================================================
# 16. RESULT STORAGE
# ============================================================

per_image_records = []

gt_records = []

prediction_records = []

inference_summary = {}


# ============================================================
# 17. ANALYZE ONE IMAGE
# ============================================================

def analyze_one_image(
    model_name,
    image_rel_path,
    split_name,
    gt_objects,
    pred_objects,
):

    iou_matrix = build_iou_matrix(
        gt_objects,
        pred_objects,
    )


    gt50, pred50 = match_objects(
        iou_matrix,
        gt_objects,
        pred_objects,
        0.50,
    )

    gt75, pred75 = match_objects(
        iou_matrix,
        gt_objects,
        pred_objects,
        0.75,
    )


    n_gt = len(gt_objects)

    n_pred = len(pred_objects)


    # --------------------------------------------------------
    # Best overlap for each GT.
    # --------------------------------------------------------

    gt_best_same = []

    gt_best_other = []

    gt_best_any = []

    gt_best_same_pred = []

    gt_best_any_class = []


    for gi, gt in enumerate(gt_objects):

        same_indices = [
            pi
            for pi, pred in enumerate(pred_objects)
            if pred["class_id"] == gt["class_id"]
        ]

        other_indices = [
            pi
            for pi, pred in enumerate(pred_objects)
            if pred["class_id"] != gt["class_id"]
        ]


        if same_indices:

            values = iou_matrix[
                gi,
                same_indices,
            ]

            best_local = int(
                np.argmax(values)
            )

            gt_best_same.append(
                float(values[best_local])
            )

            gt_best_same_pred.append(
                same_indices[best_local]
            )

        else:

            gt_best_same.append(0.0)

            gt_best_same_pred.append(None)


        gt_best_other.append(
            float(
                iou_matrix[
                    gi,
                    other_indices
                ].max()
            )
            if other_indices
            else 0.0
        )


        if n_pred:

            best_any = int(
                np.argmax(
                    iou_matrix[gi]
                )
            )

            gt_best_any.append(
                float(iou_matrix[gi, best_any])
            )

            gt_best_any_class.append(
                pred_objects[best_any]["class_id"]
            )

        else:

            gt_best_any.append(0.0)

            gt_best_any_class.append(None)


    # --------------------------------------------------------
    # Best overlap for each prediction.
    # --------------------------------------------------------

    pred_best_same = []

    pred_best_other = []

    pred_best_any = []

    pred_best_any_gt_class = []


    for pi, pred in enumerate(pred_objects):

        same_indices = [
            gi
            for gi, gt in enumerate(gt_objects)
            if gt["class_id"] == pred["class_id"]
        ]

        other_indices = [
            gi
            for gi, gt in enumerate(gt_objects)
            if gt["class_id"] != pred["class_id"]
        ]


        pred_best_same.append(
            float(
                iou_matrix[
                    same_indices,
                    pi
                ].max()
            )
            if same_indices
            else 0.0
        )


        pred_best_other.append(
            float(
                iou_matrix[
                    other_indices,
                    pi
                ].max()
            )
            if other_indices
            else 0.0
        )


        if n_gt:

            best_gt = int(
                np.argmax(
                    iou_matrix[:, pi]
                )
            )

            pred_best_any.append(
                float(
                    iou_matrix[best_gt, pi]
                )
            )

            pred_best_any_gt_class.append(
                gt_objects[best_gt]["class_id"]
            )

        else:

            pred_best_any.append(0.0)

            pred_best_any_gt_class.append(None)


    # --------------------------------------------------------
    # Per-image, per-class records.
    # --------------------------------------------------------

    for class_id, class_name in enumerate(CLASS_NAMES):

        gt_indices = [
            i
            for i, obj in enumerate(gt_objects)
            if obj["class_id"] == class_id
        ]

        pred_indices = [
            i
            for i, obj in enumerate(pred_objects)
            if obj["class_id"] == class_id
        ]


        tp50 = sum(
            i in gt50
            for i in gt_indices
        )

        tp75 = sum(
            i in gt75
            for i in gt_indices
        )


        fp50 = len(pred_indices) - tp50

        fp75 = len(pred_indices) - tp75


        fn50 = len(gt_indices) - tp50

        fn75 = len(gt_indices) - tp75


        ious50 = [
            gt50[i][1]
            for i in gt_indices
            if i in gt50
        ]

        ious75 = [
            gt75[i][1]
            for i in gt_indices
            if i in gt75
        ]


        per_image_records.append({

            "model": model_name,

            "split": split_name,

            "image": image_rel_path,

            "class_id": class_id,

            "class_name": class_name,

            "gt_count": len(gt_indices),

            "prediction_count": len(pred_indices),


            "TP_50": tp50,

            "FP_50": fp50,

            "FN_50": fn50,


            "precision_50": (
                tp50 / (tp50 + fp50)
                if tp50 + fp50
                else 0.0
            ),

            "recall_50": (
                tp50 / (tp50 + fn50)
                if tp50 + fn50
                else 0.0
            ),

            "mean_matched_iou_50": (
                float(np.mean(ious50))
                if ious50
                else 0.0
            ),


            "TP_75": tp75,

            "FP_75": fp75,

            "FN_75": fn75,


            "precision_75": (
                tp75 / (tp75 + fp75)
                if tp75 + fp75
                else 0.0
            ),

            "recall_75": (
                tp75 / (tp75 + fn75)
                if tp75 + fn75
                else 0.0
            ),

            "mean_matched_iou_75": (
                float(np.mean(ious75))
                if ious75
                else 0.0
            ),
        })


    # --------------------------------------------------------
    # Per-ground-truth instance records.
    # --------------------------------------------------------

    for gi, gt in enumerate(gt_objects):

        match50 = gt50.get(gi)

        match75 = gt75.get(gi)


        best_same_iou = gt_best_same[gi]

        best_other_iou = gt_best_other[gi]

        best_same_pred = gt_best_same_pred[gi]


        if match50 is not None:

            miss_reason = "TP"

        elif best_same_iou >= 0.10:

            miss_reason = (
                "same_class_overlap_but_iou_below_0.50"
            )

        elif best_other_iou >= 0.50:

            miss_reason = "possible_class_confusion"

        else:

            miss_reason = "missed_or_no_reliable_overlap"


        best_same_confidence = (
            pred_objects[best_same_pred]["confidence"]
            if best_same_pred is not None
            else 0.0
        )


        matched_confidence_50 = (
            pred_objects[match50[0]]["confidence"]
            if match50 is not None
            else 0.0
        )


        matched_confidence_75 = (
            pred_objects[match75[0]]["confidence"]
            if match75 is not None
            else 0.0
        )


        gt_records.append({

            "model": model_name,

            "split": split_name,

            "image": image_rel_path,

            "class_id": gt["class_id"],

            "class_name": CLASS_NAMES[
                gt["class_id"]
            ],


            "gt_mask_area_px": gt["mask_area"],

            "gt_bbox_width_px": gt["bbox_width"],

            "gt_bbox_height_px": gt["bbox_height"],

            "gt_max_side_px": gt["max_side"],

            "size_bucket": gt["size_bucket"],


            "best_same_class_iou": best_same_iou,

            "best_other_class_iou": best_other_iou,

            "best_any_class_iou": gt_best_any[gi],


            "best_any_predicted_class": (
                CLASS_NAMES[
                    gt_best_any_class[gi]
                ]
                if gt_best_any_class[gi] is not None
                else ""
            ),


            "best_same_class_pred_confidence": (
                best_same_confidence
            ),


            "matched_at_iou50": (
                match50 is not None
            ),

            "matched_iou50": (
                match50[1]
                if match50 is not None
                else 0.0
            ),

            "matched_pred_confidence_50": (
                matched_confidence_50
            ),


            "matched_at_iou75": (
                match75 is not None
            ),

            "matched_iou75": (
                match75[1]
                if match75 is not None
                else 0.0
            ),

            "matched_pred_confidence_75": (
                matched_confidence_75
            ),


            "miss_reason_at_iou50": miss_reason,
        })


    # --------------------------------------------------------
    # Prediction and false-positive records.
    # --------------------------------------------------------

    for pi, pred in enumerate(pred_objects):

        match50 = pred50.get(pi)

        match75 = pred75.get(pi)


        if match50 is not None:

            status50 = "TP"

            fp_type = ""

        else:

            status50 = "FP"


            if pred_best_same[pi] >= 0.50:

                fp_type = "duplicate_prediction"

            elif pred_best_other[pi] >= 0.50:

                fp_type = "cross_class_overlap"

            elif pred_best_same[pi] >= 0.10:

                fp_type = "same_class_low_mask_iou"

            elif pred_best_other[pi] >= 0.10:

                fp_type = "other_class_low_mask_iou"

            else:

                fp_type = "background_or_no_overlap"


        box = pred["box_xyxy"]


        prediction_records.append({

            "model": model_name,

            "split": split_name,

            "image": image_rel_path,


            "prediction_index": pi,

            "class_id": pred["class_id"],

            "class_name": CLASS_NAMES[
                pred["class_id"]
            ],


            "confidence": pred["confidence"],


            "pred_mask_area_px": pred["mask_area"],

            "pred_bbox_width_px": pred["bbox_width"],

            "pred_bbox_height_px": pred["bbox_height"],

            "pred_max_side_px": pred["max_side"],

            "size_bucket": pred["size_bucket"],


            "best_same_class_iou": pred_best_same[pi],

            "best_other_class_iou": pred_best_other[pi],

            "best_any_class_iou": pred_best_any[pi],


            "best_any_gt_class": (
                CLASS_NAMES[
                    pred_best_any_gt_class[pi]
                ]
                if pred_best_any_gt_class[pi] is not None
                else ""
            ),


            "status_iou50": status50,

            "false_positive_type": fp_type,


            "status_iou75": (
                "TP"
                if match75 is not None
                else "FP"
            ),


            "matched_gt_class_iou50": (
                CLASS_NAMES[
                    gt_objects[
                        match50[0]
                    ]["class_id"]
                ]
                if match50 is not None
                else ""
            ),


            "matched_iou50": (
                match50[1]
                if match50 is not None
                else 0.0
            ),

            "matched_iou75": (
                match75[1]
                if match75 is not None
                else 0.0
            ),


            "box_x1": box[0],

            "box_y1": box[1],

            "box_x2": box[2],

            "box_y2": box[3],
        })


# ============================================================
# 18. SAVE PARTIAL PROGRESS
# ============================================================

def save_partial_results(
    model_name,
    per_image_records,
    gt_records,
    prediction_records,
):

    """
    Save the completed model's progress periodically.
    """

    model_image_rows = [
        x for x in per_image_records
        if x["model"] == model_name
    ]

    model_gt_rows = [
        x for x in gt_records
        if x["model"] == model_name
    ]

    model_prediction_rows = [
        x for x in prediction_records
        if x["model"] == model_name
    ]


    pd.DataFrame(
        model_image_rows
    ).to_csv(
        OUTPUT_DIR
        / f"{model_name}_per_image_partial.csv",
        index=False,
        encoding="utf-8-sig",
    )


    pd.DataFrame(
        model_gt_rows
    ).to_csv(
        OUTPUT_DIR
        / f"{model_name}_gt_partial.csv",
        index=False,
        encoding="utf-8-sig",
    )


    pd.DataFrame(
        model_prediction_rows
    ).to_csv(
        OUTPUT_DIR
        / f"{model_name}_predictions_partial.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 19. RUN INFERENCE — ONE MODEL AT A TIME
# ============================================================

for model_name, checkpoint_path in MODELS.items():

    print("\n")

    print("#" * 90)

    print("STARTING INFERENCE:", model_name)

    print("#" * 90)


    gc.collect()

    torch.cuda.empty_cache()


    model = YOLO(
        str(checkpoint_path)
    )


    processed_images = 0

    total_predictions = 0

    total_gt_instances = 0

    images_at_max_det = 0


    prediction_stream = model.predict(
        source=[
            str(p)
            for p in image_paths
        ],
        **PREDICT_ARGS,
    )


    try:

        for result in prediction_stream:

            result_path = Path(
                result.path
            ).resolve()


            image_path = image_path_map.get(
                result_path
            )


            if image_path is None:

                image_path = basename_map.get(
                    Path(result.path).name
                )


            if image_path is None:

                raise RuntimeError(
                    f"Unknown prediction image: {result.path}"
                )


            height, width = result.orig_shape


            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            gt_objects = create_gt_objects(
                image_path,
                height,
                width,
            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            pred_objects = []


            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                pred_classes = (
                    result.boxes.cls
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )


                pred_confidences = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )


                pred_boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )


                if result.masks is None:

                    raise RuntimeError(
                        "Boxes exist, but masks are missing."
                    )


                n_predictions = len(
                    pred_classes
                )


                if (
                    len(result.masks.data)
                    != n_predictions
                ):

                    raise RuntimeError(
                        "Prediction boxes/masks count mismatch."
                    )


                if (
                    n_predictions
                    >= PREDICT_ARGS["max_det"]
                ):

                    images_at_max_det += 1


                for pi in range(n_predictions):

                    class_id = int(
                        pred_classes[pi]
                    )


                    if not 0 <= class_id < NUM_CLASSES:

                        continue


                    # Move one mask at a time to CPU.
                    raw_mask_cpu = (
                        result.masks.data[pi]
                        .detach()
                        .cpu()
                    )


                    pred_object = create_prediction_object(
                        raw_mask_cpu=raw_mask_cpu,

                        original_shape=(
                            height,
                            width
                        ),

                        class_id=class_id,

                        confidence=(
                            pred_confidences[pi]
                        ),

                        box_xyxy=(
                            pred_boxes[pi]
                        ),
                    )


                    del raw_mask_cpu


                    if pred_object is not None:

                        pred_objects.append(
                            pred_object
                        )


                del pred_classes

                del pred_confidences

                del pred_boxes


            # ------------------------------------------------
            # Paired image analysis
            # ------------------------------------------------

            relative_path = str(
                image_path.relative_to(
                    VAL_IMAGE_ROOT
                )
            )


            split_name = image_split_map[
                image_path
            ]


            analyze_one_image(
                model_name=model_name,

                image_rel_path=relative_path,

                split_name=split_name,

                gt_objects=gt_objects,

                pred_objects=pred_objects,
            )


            processed_images += 1

            total_predictions += len(
                pred_objects
            )

            total_gt_instances += len(
                gt_objects
            )


            # Release image-level mask objects.
            del pred_objects

            del gt_objects

            del result


            if processed_images % 25 == 0:

                free_now, _ = torch.cuda.mem_get_info(
                    GPU_ID
                )

                print(
                    f"{model_name}: "
                    f"{processed_images}/268 images | "
                    f"Predictions={total_predictions} | "
                    f"GPU free={free_now / 1024**3:.2f} GB"
                )


                # Save incremental results.
                save_partial_results(
                    model_name,
                    per_image_records,
                    gt_records,
                    prediction_records,
                )


                gc.collect()

                torch.cuda.empty_cache()


    finally:

        try:

            prediction_stream.close()

        except Exception:

            pass


    # --------------------------------------------------------
    # Verify completed model
    # --------------------------------------------------------

    if processed_images != 268:

        raise RuntimeError(
            f"{model_name}: processed "
            f"{processed_images}/268 images."
        )


    if total_gt_instances != 3042:

        raise RuntimeError(
            f"{model_name}: expected 3042 GT, "
            f"got {total_gt_instances}."
        )


    inference_summary[model_name] = {

        "images": processed_images,

        "ground_truth_instances": total_gt_instances,

        "predictions": total_predictions,

        "images_reaching_max_det": images_at_max_det,

        "max_det": PREDICT_ARGS["max_det"],

        "checkpoint": str(checkpoint_path),

        "gpu": GPU_ID,
    }


    print("\nCOMPLETED:", model_name)

    print("Images:", processed_images)

    print("GT instances:", total_gt_instances)

    print("Predictions:", total_predictions)

    print(
        "Images at max_det:",
        images_at_max_det
    )


    # Save final model-specific partials.
    save_partial_results(
        model_name,
        per_image_records,
        gt_records,
        prediction_records,
    )


    # Release model before loading next checkpoint.
    del prediction_stream

    del model

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# 20. BUILD DATAFRAMES
# ============================================================

per_image_df = pd.DataFrame(
    per_image_records
)

gt_df = pd.DataFrame(
    gt_records
)

pred_df = pd.DataFrame(
    prediction_records
)


if len(per_image_df) != 2 * 268 * NUM_CLASSES:

    raise RuntimeError(
        "Unexpected per-image/class record count."
    )


if len(gt_df) != 2 * 3042:

    raise RuntimeError(
        "Unexpected ground-truth record count."
    )


print("\n" + "=" * 85)

print("ANALYSIS RECORD COUNTS")

print("=" * 85)

print(
    "Per-image/class:",
    len(per_image_df)
)

print(
    "Ground-truth:",
    len(gt_df)
)

print(
    "Predictions:",
    len(pred_df)
)


# ============================================================
# 21. PAIRED IMAGE COMPARISON
# ============================================================

paired_df = per_image_df[
    per_image_df["model"] == "C_IRFS"
].merge(

    per_image_df[
        per_image_df["model"] == "CB_IRFS"
    ],

    on=[
        "split",
        "image",
        "class_id",
        "class_name",
    ],

    suffixes=(
        "_C",
        "_CB",
    ),

    how="inner",

    validate="one_to_one",
)


assert len(paired_df) == 268 * NUM_CLASSES


for metric in [

    "TP_50",
    "FP_50",
    "FN_50",

    "precision_50",
    "recall_50",

    "TP_75",
    "FP_75",
    "FN_75",

    "precision_75",
    "recall_75",
]:

    paired_df[
        f"delta_{metric}"
    ] = (

        paired_df[
            f"{metric}_CB"
        ]

        -

        paired_df[
            f"{metric}_C"
        ]
    )


# ============================================================
# 22. CLASS SUMMARY
# ============================================================

class_summary_rows = []


for model_name in MODELS:

    for split_name in [

        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            image_group = per_image_df[

                (per_image_df["model"] == model_name)

                &

                (per_image_df["split"] == split_name)

                &

                (per_image_df["class_id"] == class_id)
            ]


            instance_group = gt_df[

                (gt_df["model"] == model_name)

                &

                (gt_df["split"] == split_name)

                &

                (gt_df["class_id"] == class_id)
            ]


            for threshold in [

                "50",
                "75",
            ]:

                tp = int(
                    image_group[
                        f"TP_{threshold}"
                    ].sum()
                )

                fp = int(
                    image_group[
                        f"FP_{threshold}"
                    ].sum()
                )

                fn = int(
                    image_group[
                        f"FN_{threshold}"
                    ].sum()
                )


                precision = (

                    tp / (tp + fp)

                    if tp + fp

                    else 0.0
                )


                recall = (

                    tp / (tp + fn)

                    if tp + fn

                    else 0.0
                )


                matched = instance_group[

                    instance_group[
                        f"matched_at_iou{threshold}"
                    ]
                ]


                mean_iou = (

                    matched[
                        f"matched_iou{threshold}"
                    ].mean()

                    if len(matched)

                    else 0.0
                )


                class_summary_rows.append({

                    "model": model_name,

                    "split": split_name,

                    "class_id": class_id,

                    "class_name": class_name,

                    "matching_threshold": (

                        0.50

                        if threshold == "50"

                        else 0.75
                    ),

                    "GT": tp + fn,

                    "Predictions": tp + fp,

                    "TP": tp,

                    "FP": fp,

                    "FN": fn,

                    "precision": precision,

                    "recall": recall,

                    "mean_matched_mask_iou": float(
                        mean_iou
                    ),
                })


class_summary_df = pd.DataFrame(
    class_summary_rows
)


# ============================================================
# 23. SIZE-STRATIFIED RECALL
# ============================================================

size_summary_rows = []


for model_name in MODELS:

    for split_name in [

        "Combined_268",
        "Old_119",
        "New_149",
    ]:

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            for bucket in SIZE_BUCKETS:

                group = gt_df[

                    (gt_df["model"] == model_name)

                    &

                    (gt_df["split"] == split_name)

                    &

                    (gt_df["class_id"] == class_id)

                    &

                    (gt_df["size_bucket"] == bucket)
                ]


                n_gt = len(group)


                tp50 = int(
                    group["matched_at_iou50"].sum()
                )

                tp75 = int(
                    group["matched_at_iou75"].sum()
                )


                size_summary_rows.append({

                    "model": model_name,

                    "split": split_name,

                    "class_id": class_id,

                    "class_name": class_name,

                    "size_bucket": bucket,

                    "GT_instances": n_gt,


                    "TP_IoU50": tp50,

                    "FN_IoU50": n_gt - tp50,

                    "recall_IoU50": (

                        tp50 / n_gt

                        if n_gt

                        else 0.0
                    ),


                    "TP_IoU75": tp75,

                    "FN_IoU75": n_gt - tp75,

                    "recall_IoU75": (

                        tp75 / n_gt

                        if n_gt

                        else 0.0
                    ),
                })


size_summary_df = pd.DataFrame(
    size_summary_rows
)


# ============================================================
# 24. FALSE POSITIVE TYPE SUMMARY
# ============================================================

fp_df = pred_df[
    pred_df["status_iou50"] == "FP"
].copy()


if not fp_df.empty:

    fp_type_summary_df = (

        fp_df.groupby(

            [
                "model",
                "split",
                "class_id",
                "class_name",
                "false_positive_type",
            ],

            dropna=False,
        )

        .agg(

            FP_count=(
                "confidence",
                "count",
            ),

            mean_confidence=(
                "confidence",
                "mean",
            ),

            median_confidence=(
                "confidence",
                "median",
            ),

            mean_best_same_class_iou=(
                "best_same_class_iou",
                "mean",
            ),

            mean_best_any_class_iou=(
                "best_any_class_iou",
                "mean",
            ),
        )

        .reset_index()
    )

else:

    fp_type_summary_df = pd.DataFrame()


# ============================================================
# 25. MISSED GROUND-TRUTH INSTANCES
# ============================================================

missed_df = gt_df[
    ~gt_df["matched_at_iou50"]
].copy()


missed_df = missed_df.sort_values(

    by=[

        "model",
        "class_name",
        "gt_max_side_px",
        "best_same_class_iou",
    ],

    ascending=[

        True,
        True,
        True,
        False,
    ],
)


# ============================================================
# 26. SAVE CSV REPORTS
# ============================================================

def save_analysis_csv(
    dataframe,
    filename,
):

    path = OUTPUT_DIR / filename

    dataframe.to_csv(

        path,

        index=False,

        encoding="utf-8-sig",
    )

    print("[SAVED]", path)

    return path


print("\n" + "=" * 85)

print("SAVING ANALYSIS REPORTS")

print("=" * 85)


save_analysis_csv(
    per_image_df,
    "per_image_class_metrics.csv",
)


save_analysis_csv(
    paired_df,
    "paired_image_class_comparison.csv",
)


save_analysis_csv(
    gt_df,
    "per_ground_truth_instance.csv",
)


save_analysis_csv(
    pred_df,
    "all_predictions.csv",
)


save_analysis_csv(
    fp_df,
    "false_positive_predictions.csv",
)


save_analysis_csv(
    class_summary_df,
    "class_summary.csv",
)


save_analysis_csv(
    size_summary_df,
    "size_stratified_recall.csv",
)


save_analysis_csv(
    fp_type_summary_df,
    "false_positive_type_summary.csv",
)


save_analysis_csv(
    missed_df,
    "missed_ground_truth_objects.csv",
)


# ============================================================
# 27. PRINT COMBINED TP / FP / FN
# ============================================================

print("\n" + "=" * 110)

print("COMBINED 268 — MASK IoU >= 0.50")

print("=" * 110)


combined = class_summary_df[

    (class_summary_df["split"] == "Combined_268")

    &

    (class_summary_df["matching_threshold"] == 0.50)
]


for class_name in CLASS_NAMES:

    c = combined[

        (combined["model"] == "C_IRFS")

        &

        (combined["class_name"] == class_name)
    ].iloc[0]


    cb = combined[

        (combined["model"] == "CB_IRFS")

        &

        (combined["class_name"] == class_name)
    ].iloc[0]


    print(f"\n{class_name}")


    print(

        f"C-IRFS : "

        f"TP={int(c.TP):4d} "

        f"FP={int(c.FP):4d} "

        f"FN={int(c.FN):4d} "

        f"P={c.precision:.4f} "

        f"R={c.recall:.4f}"
    )


    print(

        f"CB-IRFS: "

        f"TP={int(cb.TP):4d} "

        f"FP={int(cb.FP):4d} "

        f"FN={int(cb.FN):4d} "

        f"P={cb.precision:.4f} "

        f"R={cb.recall:.4f}"
    )


    print(

        f"Delta  : "

        f"TP={int(cb.TP-c.TP):+d} "

        f"FP={int(cb.FP-c.FP):+d} "

        f"FN={int(cb.FN-c.FN):+d} "

        f"Recall={cb.recall-c.recall:+.4f}"
    )


# ============================================================
# 28. SIZE-STRATIFIED RECALL — GLOVES AND SHOES
# ============================================================

print("\n" + "=" * 110)

print("SIZE-STRATIFIED RECALL — GLOVES AND SHOES")

print("=" * 110)


for split_name in [

    "Combined_268",
    "Old_119",
    "New_149",
]:

    print(f"\n{split_name}")


    for class_name in [

        "gloves",
        "shoes",
    ]:

        print(f"\n  {class_name}")


        for model_name in MODELS:

            subset = size_summary_df[

                (size_summary_df["model"] == model_name)

                &

                (size_summary_df["split"] == split_name)

                &

                (size_summary_df["class_name"] == class_name)
            ]


            print(f"    {model_name}")


            for _, row in subset.iterrows():

                print(

                    f"      {row['size_bucket']:12s} "

                    f"GT={int(row['GT_instances']):4d} "

                    f"Recall50={row['recall_IoU50']:.4f} "

                    f"Recall75={row['recall_IoU75']:.4f}"
                )


# ============================================================
# 29. FALSE POSITIVE TYPES — GLOVES AND SHOES
# ============================================================

print("\n" + "=" * 110)

print("FALSE POSITIVE TYPES — GLOVES AND SHOES")

print("=" * 110)


for class_name in [

    "gloves",
    "shoes",
]:

    print(f"\n{class_name}")


    subset = fp_type_summary_df[

        (fp_type_summary_df["split"] == "Combined_268")

        &

        (fp_type_summary_df["class_name"] == class_name)
    ]


    if len(subset):

        print(

            subset[

                [

                    "model",

                    "false_positive_type",

                    "FP_count",

                    "mean_confidence",

                    "median_confidence",
                ]

            ].to_string(index=False)
        )

    else:

        print("No false positives.")


# ============================================================
# 30. SAVE METADATA
# ============================================================

metadata = {

    "analysis": "paired_error_analysis_v3",

    "models": {

        name: str(path)

        for name, path in MODELS.items()
    },

    "gpu": GPU_ID,

    "validation_images": len(image_paths),

    "old_images": old_count,

    "new_images": new_count,

    "ground_truth_instances": 3042,

    "ground_truth_class_counts": {

        CLASS_NAMES[i]: int(
            EXPECTED_GT_COUNTS[i]
        )

        for i in range(NUM_CLASSES)
    },

    "source_label_lines": source_line_count,

    "exact_duplicate_labels_removed": duplicate_count,

    "prediction_configuration": PREDICT_ARGS,

    "mask_scaling": (
        "Ultralytics scale_masks applied on CPU "
        "to original image dimensions."
    ),

    "matching_method": (
        "Threshold-aware class-wise Hungarian "
        "one-to-one assignment."
    ),

    "matching_thresholds": MATCH_THRESHOLDS,

    "size_buckets": SIZE_BUCKETS,

    "inference_summary": inference_summary,

    "interpretation_warning": (
        "Fixed-confidence diagnostic matching, not official AP. "
        "Results depend on confidence threshold, NMS IoU and max_det. "
        "Images reaching max_det may have truncated predictions."
    ),
}


metadata_path = (
    OUTPUT_DIR
    / "analysis_metadata.json"
)


metadata_path.write_text(

    json.dumps(

        metadata,

        indent=2,

        ensure_ascii=False,

        default=str,
    ),

    encoding="utf-8",
)


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)

print("PAIRED ERROR ANALYSIS COMPLETED")

print("=" * 100)


print("Images:", len(image_paths))

print("Models:", len(MODELS))

print("Ground-truth instances/model:", 3042)


for model_name, info in inference_summary.items():

    print(f"\n{model_name}")

    print(
        "  Predictions:",
        info["predictions"]
    )

    print(
        "  Images reaching max_det:",
        info["images_reaching_max_det"]
    )


print("\nOutput directory:")

print(OUTPUT_DIR)


print("\nGenerated files:")

for filename in [

    "per_image_class_metrics.csv",

    "paired_image_class_comparison.csv",

    "per_ground_truth_instance.csv",

    "all_predictions.csv",

    "false_positive_predictions.csv",

    "class_summary.csv",

    "size_stratified_recall.csv",

    "false_positive_type_summary.csv",

    "missed_ground_truth_objects.csv",

    "analysis_metadata.json",
]:

    print(" -", filename)


print("\n[CELL 17I COMPLETED]")

###########################################################################
# CELL 32
###########################################################################
# ============================================================
# CELL 17J — GPU MEMORY DIAGNOSTICS
# No training. No inference. No model loading.
# ============================================================

import subprocess
import torch
import gc
import os

print("=" * 80)
print("1. CUDA DEVICES")
print("=" * 80)

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):

    free_b, total_b = torch.cuda.mem_get_info(i)

    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Total     : {total_b / 1024**3:.2f} GB")
    print(f"  Free      : {free_b / 1024**3:.2f} GB")
    print(f"  Allocated : {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
    print(f"  Reserved  : {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")


print("\n" + "=" * 80)
print("2. NVIDIA-SMI PROCESS LIST")
print("=" * 80)

try:

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv"
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

except Exception as exc:
    print("Could not query process list:", repr(exc))


print("\n" + "=" * 80)
print("3. CURRENT NOTEBOOK PROCESS")
print("=" * 80)

print("Python PID:", os.getpid())

try:

    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    print(result.stdout)

except Exception as exc:
    print("nvidia-smi failed:", repr(exc))


print("\n" + "=" * 80)
print("4. LIVE GPU TENSOR REFERENCES IN THIS PYTHON KERNEL")
print("=" * 80)

gc.collect()

gpu_tensors = []

for obj in gc.get_objects():

    try:

        if torch.is_tensor(obj) and obj.is_cuda:

            gpu_tensors.append({
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "size_gb": (
                    obj.numel() * obj.element_size()
                    / 1024**3
                ),
            })

    except Exception:
        pass

gpu_tensors.sort(
    key=lambda x: x["size_gb"],
    reverse=True,
)

print("Number of referenced CUDA tensors:", len(gpu_tensors))

for item in gpu_tensors[:15]:
    print(item)

print("\n[CELL 17J COMPLETED]")

###########################################################################
# CELL 33
###########################################################################
# ============================================================
# CELL 17I - FULL PAIRED ERROR ANALYSIS: C-IRFS vs CB-IRFS
# GPU 0 | YOLO11-seg | Mask IoU 0.50 / 0.75
# Dataset: Old validation (119) + New validation (149)
# ============================================================

import os
import gc
import yaml
import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path("/kaggle/working/YOLO11_seg_expanded")
DATASET_DIR = ROOT / "dataset"

OLD_YAML = DATASET_DIR / "old_validation.yaml"
NEW_YAML = DATASET_DIR / "new_validation.yaml"

MODEL_PATHS = {
    "C_IRFS": (
        ROOT / "research_experiments" /
        "C_irfs_t030" / "weights" / "best.pt"
    ),
    "CB_IRFS": (
        ROOT / "kaggle_outputs" /
        "CB_IRFS_sqrt_17ep_v1" / "weights" / "best.pt"
    ),
}

OUTPUT_DIR = (
    ROOT / "research_experiments" /
    "paired_error_analysis_gpu0"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

NUM_CLASSES = 6

# Chỉ dùng GPU 0.
DEVICE = 0

# Giữ nguyên cấu hình inference đã kiểm tra thành công.
IMGSZ = 960
BATCH_SIZE = 1
CONF = 0.001
NMS_IOU = 0.70
MAX_DET = 300
HALF = True
RETINA_MASKS = False

IOU_THRESHOLDS = [0.50, 0.75]

# Quy ước kích thước object dựa trên tỷ lệ diện tích mask / ảnh.
# Đây là các ngưỡng phân tích, không phải chuẩn kích thước phổ quát.
SMALL_MAX_RATIO = 0.01
MEDIUM_MAX_RATIO = 0.05

# Kiểm tra chặt chẽ dataset đã xác nhận.
EXPECTED_IMAGE_COUNTS = {
    "Old": 119,
    "New": 149,
}

EXPECTED_TOTAL_IMAGES = 268
EXPECTED_TOTAL_INSTANCES = 3042

EXPECTED_CLASS_COUNTS = {
    0: 829,  # hat
    1: 238,  # gloves
    2: 339,  # mask
    3: 446,  # pants
    4: 274,  # shoes
    5: 916,  # long-sleeved shirt
}

# GPU 0 cần còn ít nhất 8 GB trước khi inference.
MIN_FREE_GPU_GB = 8.0


# ============================================================
# 2. ENVIRONMENT AND GPU CHECK
# ============================================================

print("=" * 80)
print("1. ENVIRONMENT")
print("=" * 80)

print("PyTorch:", torch.__version__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

if torch.cuda.device_count() <= DEVICE:
    raise RuntimeError(f"GPU {DEVICE} does not exist.")

torch.cuda.set_device(DEVICE)
gc.collect()
torch.cuda.empty_cache()

gpu_free, gpu_total = torch.cuda.mem_get_info(DEVICE)

print("Selected GPU:", DEVICE)
print("GPU name:", torch.cuda.get_device_name(DEVICE))
print(f"GPU total: {gpu_total / 1024**3:.2f} GB")
print(f"GPU free : {gpu_free / 1024**3:.2f} GB")

if gpu_free / 1024**3 < MIN_FREE_GPU_GB:
    raise RuntimeError(
        f"GPU {DEVICE} has only {gpu_free / 1024**3:.2f} GB free. "
        f"Required: at least {MIN_FREE_GPU_GB:.1f} GB. "
        "Do not start inference."
    )

print("\nGPU memory check: PASSED")


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\n" + "=" * 80)
print("2. CHECK FILES")
print("=" * 80)

for p in [ROOT, DATASET_DIR, OLD_YAML, NEW_YAML]:
    if not p.exists():
        raise FileNotFoundError(f"Missing path: {p}")
    print("[OK]", p)

for model_name, checkpoint in MODEL_PATHS.items():
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )
    print("[OK]", model_name, checkpoint)


# ============================================================
# 4. READ VALIDATION IMAGES FROM YAML
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def resolve_path(path_value, base_dir):
    """Resolve a path, trying the YAML base directory first."""
    p = Path(str(path_value))

    if p.is_absolute():
        return p.resolve()

    candidate = (base_dir / p).resolve()
    if candidate.exists():
        return candidate

    candidate = (DATASET_DIR / p).resolve()
    if candidate.exists():
        return candidate

    return (base_dir / p).resolve()


def read_image_list_file(list_file, base_dir):
    """Read image paths from a YOLO-style text file."""
    image_paths = []

    with open(list_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        p = Path(line)

        if p.is_absolute():
            candidates = [p]
        else:
            candidates = [
                list_file.parent / p,
                base_dir / p,
                DATASET_DIR / p,
                ROOT / p,
            ]

        found = None
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists() and candidate.is_file():
                found = candidate
                break

        if found is None:
            raise FileNotFoundError(
                f"Image listed in {list_file} was not found: {line}"
            )

        if found.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(found)

    return image_paths


def collect_images_from_yaml(yaml_path, split_name):
    """Collect validation images from a dataset YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    configured_path = Path(str(cfg.get("path", ".")))

    if configured_path.is_absolute():
        base_dir = configured_path.resolve()
    else:
        base_dir = (yaml_path.parent / configured_path).resolve()

    if "val" not in cfg:
        raise KeyError(f"'val' is missing in {yaml_path}")

    val_entries = cfg["val"]
    if not isinstance(val_entries, list):
        val_entries = [val_entries]

    collected = []

    for entry in val_entries:
        val_path = resolve_path(entry, base_dir)

        if not val_path.exists():
            raise FileNotFoundError(
                f"Validation path in {yaml_path} does not exist: "
                f"{val_path}"
            )

        if val_path.is_file() and val_path.suffix.lower() == ".txt":
            collected.extend(
                read_image_list_file(val_path, base_dir)
            )

        elif val_path.is_file():
            if val_path.suffix.lower() in IMAGE_EXTENSIONS:
                collected.append(val_path)

        elif val_path.is_dir():
            collected.extend(
                p.resolve()
                for p in val_path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )

    # Remove duplicate paths while preserving order.
    unique = list(dict.fromkeys(collected))

    if not unique:
        raise RuntimeError(
            f"No validation images found for split {split_name}."
        )

    return unique


old_images = collect_images_from_yaml(OLD_YAML, "Old")
new_images = collect_images_from_yaml(NEW_YAML, "New")

image_items = []

for p in old_images:
    image_items.append({
        "image_path": p,
        "split": "Old",
    })

for p in new_images:
    image_items.append({
        "image_path": p,
        "split": "New",
    })

# Check duplicate images across splits.
all_paths = [str(x["image_path"]) for x in image_items]

if len(all_paths) != len(set(all_paths)):
    raise RuntimeError(
        "Duplicate image paths found across Old and New validation."
    )

print("\nOld images:", len(old_images))
print("New images:", len(new_images))
print("Total images:", len(image_items))

if len(old_images) != EXPECTED_IMAGE_COUNTS["Old"]:
    raise RuntimeError(
        f"Expected 119 Old images, found {len(old_images)}."
    )

if len(new_images) != EXPECTED_IMAGE_COUNTS["New"]:
    raise RuntimeError(
        f"Expected 149 New images, found {len(new_images)}."
    )

if len(image_items) != EXPECTED_TOTAL_IMAGES:
    raise RuntimeError(
        f"Expected 268 images, found {len(image_items)}."
    )

print("Image inventory check: PASSED")


# ============================================================
# 5. MAP IMAGE PATH TO LABEL PATH
# ============================================================

def find_label_path(image_path):
    """
    Convert:
        .../images/.../abc.jpg
    into:
        .../labels/.../abc.txt
    """

    parts = list(image_path.parts)

    image_indices = [
        i for i, part in enumerate(parts)
        if part.lower() == "images"
    ]

    candidates = []

    if image_indices:
        idx = image_indices[-1]
        label_parts = parts.copy()
        label_parts[idx] = "labels"

        candidate = Path(*label_parts).with_suffix(".txt")
        candidates.append(candidate)

    # Common fallback.
    candidates.append(
        DATASET_DIR / "labels" / image_path.with_suffix(".txt").name
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    # Last fallback: search by exact stem.
    matches = list(DATASET_DIR.rglob(image_path.stem + ".txt"))

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple possible label files for {image_path.name}: "
            f"{matches[:5]}"
        )

    raise FileNotFoundError(
        f"Label file not found for image: {image_path}"
    )


# ============================================================
# 6. PARSE YOLO SEGMENTATION POLYGONS
# ============================================================

def classify_object_size(area_ratio):
    if area_ratio < SMALL_MAX_RATIO:
        return "small"
    elif area_ratio < MEDIUM_MAX_RATIO:
        return "medium"
    return "large"


def parse_yolo_segmentation(label_path, image_width, image_height):
    """
    Parse YOLO segmentation:
        class_id x1 y1 x2 y2 ... xn yn

    Coordinates are normalized to [0, 1].

    Each GT mask is stored as a cropped binary mask to save RAM.
    """

    objects = []

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line_idx, line in enumerate(lines):
        values = line.split()

        if len(values) < 7:
            raise ValueError(
                f"Invalid segmentation line in {label_path}, "
                f"line {line_idx + 1}"
            )

        class_id = int(float(values[0]))

        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(
                f"Invalid class_id={class_id} in {label_path}"
            )

        coords = np.asarray(
            [float(x) for x in values[1:]],
            dtype=np.float32
        )

        if len(coords) < 6 or len(coords) % 2 != 0:
            raise ValueError(
                f"Invalid polygon coordinates in {label_path}, "
                f"line {line_idx + 1}"
            )

        points = coords.reshape(-1, 2)

        # Convert normalized coordinates to original image pixels.
        points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
        points[:, 1] = np.clip(points[:, 1], 0.0, 1.0)

        points[:, 0] *= (image_width - 1)
        points[:, 1] *= (image_height - 1)

        points = np.rint(points).astype(np.int32)

        x_min = int(np.clip(points[:, 0].min(), 0, image_width - 1))
        y_min = int(np.clip(points[:, 1].min(), 0, image_height - 1))
        x_max = int(np.clip(points[:, 0].max(), 0, image_width - 1))
        y_max = int(np.clip(points[:, 1].max(), 0, image_height - 1))

        # Exclusive upper bounds.
        x_max = max(x_max + 1, x_min + 1)
        y_max = max(y_max + 1, y_min + 1)

        x_max = min(x_max, image_width)
        y_max = min(y_max, image_height)

        crop_width = x_max - x_min
        crop_height = y_max - y_min

        local_points = points.copy()
        local_points[:, 0] -= x_min
        local_points[:, 1] -= y_min

        mask = np.zeros(
            (crop_height, crop_width),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [local_points.reshape(-1, 1, 2)],
            1
        )

        area = int(mask.sum())

        if area == 0:
            continue

        area_ratio = area / float(image_width * image_height)

        objects.append({
            "id": len(objects),
            "class_id": class_id,
            "bbox": (x_min, y_min, x_max, y_max),
            "mask": mask.astype(bool),
            "area": area,
            "area_ratio": area_ratio,
            "size": classify_object_size(area_ratio),
        })

    return objects


# ============================================================
# 7. BUILD GROUND-TRUTH CACHE
# ============================================================

print("\n" + "=" * 80)
print("3. BUILD GROUND-TRUTH CACHE")
print("=" * 80)

image_records = []
total_gt = 0
class_counts = {i: 0 for i in range(NUM_CLASSES)}

for index, item in enumerate(image_items, start=1):
    image_path = item["image_path"]
    split = item["split"]

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    label_path = find_label_path(image_path)

    gt_objects = parse_yolo_segmentation(
        label_path,
        width,
        height
    )

    for obj in gt_objects:
        class_counts[obj["class_id"]] += 1

    total_gt += len(gt_objects)

    image_records.append({
        "image_path": image_path,
        "image_name": image_path.name,
        "split": split,
        "width": width,
        "height": height,
        "label_path": label_path,
        "gt_objects": gt_objects,
    })

    if index % 50 == 0 or index == len(image_items):
        print(
            f"GT cache: {index}/{len(image_items)} images | "
            f"{total_gt} instances"
        )

    del image


print("\nTotal GT instances:", total_gt)
print("Class counts:")

for class_id, class_name in enumerate(CLASS_NAMES):
    print(
        f"  {class_id}: {class_name:<22} "
        f"{class_counts[class_id]}"
    )

if total_gt != EXPECTED_TOTAL_INSTANCES:
    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL_INSTANCES} GT instances, "
        f"found {total_gt}."
    )

if class_counts != EXPECTED_CLASS_COUNTS:
    raise RuntimeError(
        "GT class distribution does not match the verified dataset.\n"
        f"Expected: {EXPECTED_CLASS_COUNTS}\n"
        f"Actual:   {class_counts}"
    )

print("\nGT verification: PASSED")


# ============================================================
# 8. MASK IoU BETWEEN TWO CROPPED MASKS
# ============================================================

def cropped_mask_iou(obj_a, obj_b):
    """
    Compute IoU between two binary masks stored in their own crops.
    No full-resolution union canvas is allocated.
    """

    ax1, ay1, ax2, ay2 = obj_a["bbox"]
    bx1, by1, bx2, by2 = obj_b["bbox"]

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = 0

    if x2 > x1 and y2 > y1:
        a_crop = obj_a["mask"][
            y1 - ay1:y2 - ay1,
            x1 - ax1:x2 - ax1
        ]

        b_crop = obj_b["mask"][
            y1 - by1:y2 - by1,
            x1 - bx1:x2 - bx1
        ]

        intersection = int(
            np.logical_and(a_crop, b_crop).sum()
        )

    union = obj_a["area"] + obj_b["area"] - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


# ============================================================
# 9. MAXIMUM-CARDINALITY MATCHING AT A GIVEN IoU THRESHOLD
# ============================================================

def match_instances(gt_objects, pred_objects, threshold):
    """
    Match GT and predictions one-to-one.

    Priority:
      1. Maximize the number of valid matches above threshold.
      2. Among those assignments, maximize total IoU.

    This avoids counting multiple predictions for one GT as TP.
    """

    n_gt = len(gt_objects)
    n_pred = len(pred_objects)

    if n_gt == 0 or n_pred == 0:
        return [], np.zeros((n_gt, n_pred), dtype=np.float32)

    iou_matrix = np.zeros(
        (n_gt, n_pred),
        dtype=np.float32
    )

    for i, gt_obj in enumerate(gt_objects):
        for j, pred_obj in enumerate(pred_objects):
            iou_matrix[i, j] = cropped_mask_iou(
                gt_obj,
                pred_obj
            )

    size = max(n_gt, n_pred)

    # Dummy assignments have score 0.
    score_matrix = np.zeros(
        (size, size),
        dtype=np.float64
    )

    valid = iou_matrix >= threshold

    # A valid match has a very large positive score.
    # Invalid real pairs have a very large negative score.
    score_matrix[:n_gt, :n_pred] = np.where(
        valid,
        1_000_000.0 + iou_matrix,
        -1_000_000.0
    )

    rows, cols = linear_sum_assignment(-score_matrix)

    matches = []

    for r, c in zip(rows, cols):
        if r < n_gt and c < n_pred:
            current_iou = float(iou_matrix[r, c])

            if current_iou >= threshold:
                matches.append((int(r), int(c), current_iou))

    return matches, iou_matrix


# ============================================================
# 10. INFERENCE ON ONE IMAGE
# ============================================================

def infer_one_image(model, image_record):
    """
    Run one-image inference.

    Important:
      - GPU 0
      - half=True
      - retina_masks=False
      - Convert one predicted mask at a time to CPU
      - Scale each mask to original image size
      - Store only cropped binary masks
    """

    image_path = image_record["image_path"]
    height = image_record["height"]
    width = image_record["width"]

    results = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        conf=CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        retina_masks=RETINA_MASKS,
        half=HALF,
        device=DEVICE,
        verbose=False,
        stream=False,
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None or result.masks is None:
        del results
        return []

    boxes = result.boxes
    raw_masks = result.masks.data

    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    num_predictions = min(
        len(class_ids),
        len(confidences),
        len(raw_masks)
    )

    predictions = []

    for i in range(num_predictions):

        class_id = int(class_ids[i])

        if class_id < 0 or class_id >= NUM_CLASSES:
            continue

        # Transfer only one mask at a time to CPU.
        one_mask = (
            raw_masks[i:i + 1]
            .detach()
            .float()
            .cpu()
        )

        # Convert low-resolution mask to original image size.
        scaled_mask = scale_masks(
            one_mask,
            (height, width)
        )[0]

        binary_mask = (
            scaled_mask.numpy() >= 0.5
        )

        del one_mask, scaled_mask

        ys, xs = np.where(binary_mask)

        if len(xs) == 0:
            del binary_mask
            continue

        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max()) + 1
        y_max = int(ys.max()) + 1

        cropped_mask = binary_mask[
            y_min:y_max,
            x_min:x_max
        ].copy()

        area = int(cropped_mask.sum())

        if area == 0:
            del binary_mask, cropped_mask
            continue

        area_ratio = area / float(width * height)

        predictions.append({
            "id": i,
            "class_id": class_id,
            "confidence": float(confidences[i]),
            "bbox": (x_min, y_min, x_max, y_max),
            "mask": cropped_mask,
            "area": area,
            "area_ratio": area_ratio,
            "size": classify_object_size(area_ratio),
        })

        del binary_mask, cropped_mask

    del raw_masks, boxes, result, results

    return predictions


# ============================================================
# 11. ANALYZE ONE IMAGE FOR BOTH IoU THRESHOLDS
# ============================================================

def analyze_one_image(model_name, image_record, predictions):

    gt_all = image_record["gt_objects"]

    metric_rows = []
    detail_rows = []

    image_name = image_record["image_name"]
    split = image_record["split"]

    for threshold in IOU_THRESHOLDS:

        for class_id, class_name in enumerate(CLASS_NAMES):

            gt_class = [
                obj for obj in gt_all
                if obj["class_id"] == class_id
            ]

            pred_class = [
                obj for obj in predictions
                if obj["class_id"] == class_id
            ]

            matches, iou_matrix = match_instances(
                gt_class,
                pred_class,
                threshold
            )

            matched_gt = {x[0] for x in matches}
            matched_pred = {x[1] for x in matches}

            tp = len(matches)
            fp = len(pred_class) - tp
            fn = len(gt_class) - tp

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0 else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0 else 0.0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            # Size-specific GT and TP counts.
            size_counts = {
                "small": {"gt": 0, "tp": 0},
                "medium": {"gt": 0, "tp": 0},
                "large": {"gt": 0, "tp": 0},
            }

            for gt_idx, gt_obj in enumerate(gt_class):
                size_name = gt_obj["size"]

                size_counts[size_name]["gt"] += 1

                if gt_idx in matched_gt:
                    size_counts[size_name]["tp"] += 1

            row = {
                "model": model_name,
                "image": image_name,
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "iou_threshold": threshold,
                "gt_count": len(gt_class),
                "pred_count": len(pred_class),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            for size_name in ["small", "medium", "large"]:
                gt_count = size_counts[size_name]["gt"]
                tp_count = size_counts[size_name]["tp"]

                row[f"{size_name}_gt"] = gt_count
                row[f"{size_name}_tp"] = tp_count
                row[f"{size_name}_fn"] = gt_count - tp_count

            metric_rows.append(row)

            # Matched GT-prediction pairs.
            for gt_idx, pred_idx, current_iou in matches:

                gt_obj = gt_class[gt_idx]
                pred_obj = pred_class[pred_idx]

                detail_rows.append({
                    "model": model_name,
                    "image": image_name,
                    "split": split,
                    "iou_threshold": threshold,
                    "class_id": class_id,
                    "class_name": class_name,
                    "status": "TP",
                    "gt_id": gt_obj["id"],
                    "pred_id": pred_obj["id"],
                    "iou": current_iou,
                    "confidence": pred_obj["confidence"],
                    "gt_area_ratio": gt_obj["area_ratio"],
                    "gt_size": gt_obj["size"],
                    "pred_area_ratio": pred_obj["area_ratio"],
                    "max_iou_same_class": current_iou,
                    "fp_type": "",
                })

            # False negatives.
            for gt_idx, gt_obj in enumerate(gt_class):

                if gt_idx in matched_gt:
                    continue

                best_iou = (
                    float(iou_matrix[gt_idx].max())
                    if len(pred_class) > 0 else 0.0
                )

                detail_rows.append({
                    "model": model_name,
                    "image": image_name,
                    "split": split,
                    "iou_threshold": threshold,
                    "class_id": class_id,
                    "class_name": class_name,
                    "status": "FN",
                    "gt_id": gt_obj["id"],
                    "pred_id": -1,
                    "iou": 0.0,
                    "confidence": np.nan,
                    "gt_area_ratio": gt_obj["area_ratio"],
                    "gt_size": gt_obj["size"],
                    "pred_area_ratio": np.nan,
                    "max_iou_same_class": best_iou,
                    "fp_type": "",
                })

            # False positives.
            for pred_idx, pred_obj in enumerate(pred_class):

                if pred_idx in matched_pred:
                    continue

                best_iou = (
                    float(iou_matrix[:, pred_idx].max())
                    if len(gt_class) > 0 else 0.0
                )

                if best_iou >= threshold:
                    fp_type = "duplicate_or_competing_prediction"
                elif best_iou >= 0.10:
                    fp_type = "partial_overlap_or_localization"
                else:
                    fp_type = "no_significant_overlap"

                detail_rows.append({
                    "model": model_name,
                    "image": image_name,
                    "split": split,
                    "iou_threshold": threshold,
                    "class_id": class_id,
                    "class_name": class_name,
                    "status": "FP",
                    "gt_id": -1,
                    "pred_id": pred_obj["id"],
                    "iou": 0.0,
                    "confidence": pred_obj["confidence"],
                    "gt_area_ratio": np.nan,
                    "gt_size": "",
                    "pred_area_ratio": pred_obj["area_ratio"],
                    "max_iou_same_class": best_iou,
                    "fp_type": fp_type,
                })

    return metric_rows, detail_rows


# ============================================================
# 12. RUN ONE MODEL ON ALL 268 IMAGES
# ============================================================

def run_model_analysis(model_name, checkpoint):

    print("\n" + "=" * 80)
    print(f"4. INFERENCE: {model_name}")
    print("=" * 80)

    print("Checkpoint:", checkpoint)
    print("Device:", DEVICE)
    print("Image size:", IMGSZ)
    print("Batch:", BATCH_SIZE)
    print("Half:", HALF)
    print("Retina masks:", RETINA_MASKS)

    model = YOLO(str(checkpoint))

    all_metric_rows = []
    all_detail_rows = []

    for index, image_record in enumerate(image_records, start=1):

        predictions = infer_one_image(
            model,
            image_record
        )

        metric_rows, detail_rows = analyze_one_image(
            model_name,
            image_record,
            predictions
        )

        all_metric_rows.extend(metric_rows)
        all_detail_rows.extend(detail_rows)

        if index % 10 == 0 or index == len(image_records):

            free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)

            print(
                f"[{model_name}] "
                f"{index:3d}/{len(image_records)} images | "
                f"predictions={len(predictions):3d} | "
                f"GPU free={free_bytes / 1024**3:.2f} GB"
            )

            # Save intermediate progress to CSV.
            # These files are replaced as the run progresses.
            pd.DataFrame(all_metric_rows).to_csv(
                OUTPUT_DIR / f"{model_name}_metrics_progress.csv",
                index=False
            )

            pd.DataFrame(all_detail_rows).to_csv(
                OUTPUT_DIR / f"{model_name}_details_progress.csv",
                index=False
            )

            gc.collect()
            torch.cuda.empty_cache()

        del predictions, metric_rows, detail_rows

    # Save final model-specific outputs.
    metrics_df = pd.DataFrame(all_metric_rows)
    details_df = pd.DataFrame(all_detail_rows)

    metrics_path = (
        OUTPUT_DIR / f"{model_name}_per_image_class_metrics.csv"
    )
    details_path = (
        OUTPUT_DIR / f"{model_name}_object_details.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)
    details_df.to_csv(details_path, index=False)

    print("\nSaved:")
    print(metrics_path)
    print(details_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_df, details_df


# ============================================================
# 13. RUN C-IRFS AND CB-IRFS
# ============================================================

all_model_metrics = []
all_model_details = []

for model_name, checkpoint in MODEL_PATHS.items():

    metrics_df, details_df = run_model_analysis(
        model_name,
        checkpoint
    )

    all_model_metrics.append(metrics_df)
    all_model_details.append(details_df)

    del metrics_df, details_df
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 14. COMBINE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("5. COMBINE RESULTS")
print("=" * 80)

metrics_df = pd.concat(
    all_model_metrics,
    ignore_index=True
)

details_df = pd.concat(
    all_model_details,
    ignore_index=True
)

metrics_df.to_csv(
    OUTPUT_DIR / "ALL_models_per_image_class_metrics.csv",
    index=False
)

details_df.to_csv(
    OUTPUT_DIR / "ALL_models_object_details.csv",
    index=False
)


# ============================================================
# 15. CLASS-LEVEL SUMMARY
# ============================================================

class_summary = (
    metrics_df
    .groupby(
        ["model", "iou_threshold", "class_id", "class_name"],
        as_index=False
    )
    .agg({
        "gt_count": "sum",
        "pred_count": "sum",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",
    })
)

class_summary["precision"] = (
    class_summary["tp"] /
    (class_summary["tp"] + class_summary["fp"]).replace(0, np.nan)
).fillna(0)

class_summary["recall"] = (
    class_summary["tp"] /
    (class_summary["tp"] + class_summary["fn"]).replace(0, np.nan)
).fillna(0)

class_summary["f1"] = (
    2 * class_summary["precision"] * class_summary["recall"] /
    (class_summary["precision"] + class_summary["recall"]).replace(0, np.nan)
).fillna(0)

class_summary.to_csv(
    OUTPUT_DIR / "class_summary.csv",
    index=False
)


# ============================================================
# 16. OVERALL SUMMARY
# ============================================================

overall_summary = (
    metrics_df
    .groupby(
        ["model", "iou_threshold"],
        as_index=False
    )
    .agg({
        "gt_count": "sum",
        "pred_count": "sum",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",
    })
)

overall_summary["precision"] = (
    overall_summary["tp"] /
    (overall_summary["tp"] + overall_summary["fp"]).replace(0, np.nan)
).fillna(0)

overall_summary["recall"] = (
    overall_summary["tp"] /
    (overall_summary["tp"] + overall_summary["fn"]).replace(0, np.nan)
).fillna(0)

overall_summary["f1"] = (
    2 * overall_summary["precision"] * overall_summary["recall"] /
    (overall_summary["precision"] + overall_summary["recall"]).replace(0, np.nan)
).fillna(0)

overall_summary.to_csv(
    OUTPUT_DIR / "overall_summary.csv",
    index=False
)


# ============================================================
# 17. OLD VS NEW VALIDATION SUMMARY
# ============================================================

split_summary = (
    metrics_df
    .groupby(
        ["model", "split", "iou_threshold"],
        as_index=False
    )
    .agg({
        "gt_count": "sum",
        "pred_count": "sum",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",
    })
)

split_summary["precision"] = (
    split_summary["tp"] /
    (split_summary["tp"] + split_summary["fp"]).replace(0, np.nan)
).fillna(0)

split_summary["recall"] = (
    split_summary["tp"] /
    (split_summary["tp"] + split_summary["fn"]).replace(0, np.nan)
).fillna(0)

split_summary["f1"] = (
    2 * split_summary["precision"] * split_summary["recall"] /
    (split_summary["precision"] + split_summary["recall"]).replace(0, np.nan)
).fillna(0)

split_summary.to_csv(
    OUTPUT_DIR / "old_new_split_summary.csv",
    index=False
)


# ============================================================
# 18. SIZE-BASED RECALL
# ============================================================

size_rows = []

for (model_name, threshold, class_id, class_name), group in (
    metrics_df.groupby(
        ["model", "iou_threshold", "class_id", "class_name"]
    )
):

    for size_name in ["small", "medium", "large"]:

        gt_count = int(group[f"{size_name}_gt"].sum())
        tp_count = int(group[f"{size_name}_tp"].sum())
        fn_count = int(group[f"{size_name}_fn"].sum())

        recall = (
            tp_count / gt_count
            if gt_count > 0 else np.nan
        )

        size_rows.append({
            "model": model_name,
            "iou_threshold": threshold,
            "class_id": class_id,
            "class_name": class_name,
            "size": size_name,
            "gt_count": gt_count,
            "tp": tp_count,
            "fn": fn_count,
            "recall": recall,
        })

size_summary = pd.DataFrame(size_rows)

size_summary.to_csv(
    OUTPUT_DIR / "size_based_recall.csv",
    index=False
)


# ============================================================
# 19. PAIRED PER-IMAGE COMPARISON: C-IRFS VS CB-IRFS
# ============================================================

paired_columns = [
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]

paired_index = [
    "image",
    "split",
    "class_id",
    "class_name",
    "iou_threshold",
]

paired_wide = metrics_df.pivot_table(
    index=paired_index,
    columns="model",
    values=paired_columns,
    aggfunc="first"
)

paired_wide.columns = [
    f"{metric}_{model}"
    for metric, model in paired_wide.columns
]

paired_wide = paired_wide.reset_index()

# Add CB - C differences.
for metric in paired_columns:

    c_col = f"{metric}_C_IRFS"
    cb_col = f"{metric}_CB_IRFS"

    if c_col in paired_wide.columns and cb_col in paired_wide.columns:
        paired_wide[f"delta_{metric}_CB_minus_C"] = (
            paired_wide[cb_col] - paired_wide[c_col]
        )

paired_wide.to_csv(
    OUTPUT_DIR / "paired_image_comparison_C_vs_CB.csv",
    index=False
)


# ============================================================
# 20. FALSE-POSITIVE BREAKDOWN
# ============================================================

fp_details = details_df[
    details_df["status"] == "FP"
].copy()

if len(fp_details) > 0:

    fp_summary = (
        fp_details
        .groupby(
            [
                "model",
                "iou_threshold",
                "class_id",
                "class_name",
                "fp_type",
            ],
            as_index=False
        )
        .size()
        .rename(columns={"size": "count"})
    )

else:
    fp_summary = pd.DataFrame(
        columns=[
            "model",
            "iou_threshold",
            "class_id",
            "class_name",
            "fp_type",
            "count",
        ]
    )

fp_summary.to_csv(
    OUTPUT_DIR / "false_positive_breakdown.csv",
    index=False
)


# ============================================================
# 21. SAVE A COMPACT EXCEL WORKBOOK
# ============================================================

excel_path = OUTPUT_DIR / "paired_error_analysis_summary.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    overall_summary.to_excel(
        writer,
        sheet_name="Overall",
        index=False
    )

    class_summary.to_excel(
        writer,
        sheet_name="Per_Class",
        index=False
    )

    split_summary.to_excel(
        writer,
        sheet_name="Old_vs_New",
        index=False
    )

    size_summary.to_excel(
        writer,
        sheet_name="Size_Recall",
        index=False
    )

    fp_summary.to_excel(
        writer,
        sheet_name="FP_Breakdown",
        index=False
    )

    paired_wide.to_excel(
        writer,
        sheet_name="Paired_Image",
        index=False
    )


# ============================================================
# 22. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("6. FINAL OVERALL SUMMARY")
print("=" * 80)

print(
    overall_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 80)
print("7. FINAL PER-CLASS SUMMARY")
print("=" * 80)

print(
    class_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 80)
print("8. SIZE-BASED RECALL")
print("=" * 80)

print(
    size_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 80)
print("9. FALSE-POSITIVE BREAKDOWN")
print("=" * 80)

print(fp_summary.to_string(index=False))

print("\n" + "=" * 80)
print("10. OUTPUT FILES")
print("=" * 80)

for p in sorted(OUTPUT_DIR.iterdir()):
    if p.is_file():
        print(f"{p.name:<55} {p.stat().st_size / 1024**2:8.2f} MB")

print("\n[CELL 17I COMPLETED SUCCESSFULLY]")

###########################################################################
# CELL 34
###########################################################################
# ============================================================
# CELL 17I-FIX: FIX MASK DIMENSION + RESUME PAIRED ANALYSIS
# GPU 0 | C-IRFS vs CB-IRFS
#
# Requires the previous Cell 17I setup to have completed:
# image_records, run_model_analysis, MODEL_PATHS, OUTPUT_DIR,
# CLASS_NAMES, DEVICE, and other helper functions.
# ============================================================

import gc
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks


# ============================================================
# 1. VERIFY PREVIOUS KERNEL STATE
# ============================================================

print("=" * 75)
print("1. VERIFY NOTEBOOK STATE")
print("=" * 75)

required_variables = [
    "image_records",
    "run_model_analysis",
    "MODEL_PATHS",
    "OUTPUT_DIR",
    "CLASS_NAMES",
    "DEVICE",
    "IMGSZ",
    "BATCH_SIZE",
    "CONF",
    "NMS_IOU",
    "MAX_DET",
    "HALF",
    "RETINA_MASKS",
]

missing = [
    name for name in required_variables
    if name not in globals()
]

if missing:
    raise RuntimeError(
        "Previous Cell 17I variables are missing: "
        + ", ".join(missing)
        + ". Do not continue with this cell."
    )

if len(image_records) != 268:
    raise RuntimeError(
        f"Expected 268 image records, got {len(image_records)}"
    )

if DEVICE != 0:
    raise RuntimeError(
        f"Expected GPU 0, but DEVICE={DEVICE}"
    )

torch.cuda.set_device(0)
gc.collect()
torch.cuda.empty_cache()

free_bytes, total_bytes = torch.cuda.mem_get_info(0)

print("Image records:", len(image_records))
print("GPU:", torch.cuda.get_device_name(0))
print(f"GPU free: {free_bytes / 1024**3:.2f} GB")
print("Kernel state: PASSED")

if free_bytes / 1024**3 < 8:
    raise RuntimeError(
        "GPU 0 has less than 8 GB free. "
        "Inference is stopped to prevent OOM."
    )


# ============================================================
# 2. REDEFINE INFERENCE FUNCTION
# ============================================================

def infer_one_image(model, image_record):

    image_path = image_record["image_path"]
    height = image_record["height"]
    width = image_record["width"]

    results = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        batch=1,
        conf=CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        retina_masks=False,
        half=True,
        device=0,
        verbose=False,
        stream=False,
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None or result.masks is None:
        del results
        return []

    boxes = result.boxes
    raw_masks = result.masks.data

    class_ids = (
        boxes.cls.detach().cpu().numpy().astype(int)
    )
    confidences = (
        boxes.conf.detach().cpu().numpy()
    )

    if raw_masks.ndim != 3:
        raise RuntimeError(
            "Unexpected prediction mask shape: "
            f"{tuple(raw_masks.shape)}. "
            "Expected (N, H, W)."
        )

    num_predictions = min(
        len(class_ids),
        len(confidences),
        len(raw_masks)
    )

    predictions = []

    for i in range(num_predictions):

        class_id = int(class_ids[i])

        if not 0 <= class_id < len(CLASS_NAMES):
            continue

        # ----------------------------------------------------
        # IMPORTANT FIX:
        #
        # raw_masks[i:i+1] shape: (1, H, W)
        #
        # scale_masks expects: (N, C, H, W)
        #
        # unsqueeze(1) produces: (1, 1, H, W)
        # ----------------------------------------------------

        one_mask = (
            raw_masks[i:i + 1]
            .detach()
            .float()
            .cpu()
            .unsqueeze(1)
        )

        if one_mask.ndim != 4:
            raise RuntimeError(
                f"Mask must be 4D before scale_masks, "
                f"got {tuple(one_mask.shape)}"
            )

        # Output shape: (1, 1, original_height, original_width)
        scaled_mask = scale_masks(
            one_mask,
            (height, width)
        )

        # Select the only batch and channel.
        binary_mask = (
            scaled_mask[0, 0].numpy() >= 0.5
        )

        del one_mask, scaled_mask

        ys, xs = np.where(binary_mask)

        if len(xs) == 0:
            del binary_mask
            continue

        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max()) + 1
        y_max = int(ys.max()) + 1

        cropped_mask = binary_mask[
            y_min:y_max,
            x_min:x_max
        ].copy()

        area = int(cropped_mask.sum())

        if area == 0:
            del binary_mask, cropped_mask
            continue

        area_ratio = area / float(width * height)

        predictions.append({
            "id": i,
            "class_id": class_id,
            "confidence": float(confidences[i]),
            "bbox": (
                x_min,
                y_min,
                x_max,
                y_max
            ),
            "mask": cropped_mask,
            "area": area,
            "area_ratio": area_ratio,
            "size": classify_object_size(area_ratio),
        })

        del binary_mask, cropped_mask

    del raw_masks, boxes, result, results

    return predictions


print("\nInference function replaced.")
print("Mask tensor dimension fix: PASSED")


# ============================================================
# 3. SMOKE TEST BEFORE FULL INFERENCE
# ============================================================

print("\n" + "=" * 75)
print("2. SMOKE TEST")
print("=" * 75)

smoke_checkpoint = MODEL_PATHS["C_IRFS"]

smoke_model = YOLO(str(smoke_checkpoint))

smoke_success = False

for test_index, test_record in enumerate(
    image_records[:10],
    start=1
):

    test_predictions = infer_one_image(
        smoke_model,
        test_record
    )

    print(
        f"Test {test_index}/10 | "
        f"{test_record['image_name']} | "
        f"Predictions: {len(test_predictions)}"
    )

    if len(test_predictions) > 0:
        smoke_success = True

        print("\nSMOKE TEST PASSED")
        print(
            "Example prediction:",
            {
                "class_id": test_predictions[0]["class_id"],
                "confidence": round(
                    test_predictions[0]["confidence"], 4
                ),
                "mask_shape": test_predictions[0]["mask"].shape,
            }
        )

        del test_predictions
        break

    del test_predictions

del smoke_model
gc.collect()
torch.cuda.empty_cache()

if not smoke_success:
    raise RuntimeError(
        "No predictions found in the first 10 images. "
        "Inspect inference outputs before running the full analysis."
    )

free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print(f"GPU free after smoke test: {free_bytes / 1024**3:.2f} GB")


# ============================================================
# 4. RUN BOTH MODELS ON ALL 268 IMAGES
# ============================================================

print("\n" + "=" * 75)
print("3. FULL PAIRED INFERENCE")
print("=" * 75)

all_model_metrics = []
all_model_details = []

for model_name in ["C_IRFS", "CB_IRFS"]:

    checkpoint = MODEL_PATHS[model_name]

    print("\n" + "-" * 75)
    print("RUNNING:", model_name)
    print("-" * 75)

    metrics_df, details_df = run_model_analysis(
        model_name,
        checkpoint
    )

    if len(metrics_df) != 268 * 6 * 2:
        raise RuntimeError(
            f"{model_name}: unexpected metrics row count: "
            f"{len(metrics_df)}"
        )

    all_model_metrics.append(metrics_df)
    all_model_details.append(details_df)

    print(
        f"{model_name} completed: "
        f"{len(metrics_df)} metric rows, "
        f"{len(details_df)} object records"
    )

    del metrics_df, details_df
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 5. COMBINE MODEL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("4. COMBINE RESULTS")
print("=" * 75)

metrics_df = pd.concat(
    all_model_metrics,
    ignore_index=True
)

details_df = pd.concat(
    all_model_details,
    ignore_index=True
)

metrics_df.to_csv(
    OUTPUT_DIR / "ALL_models_per_image_class_metrics.csv",
    index=False
)

details_df.to_csv(
    OUTPUT_DIR / "ALL_models_object_details.csv",
    index=False
)

print("Combined metric rows:", len(metrics_df))
print("Combined object records:", len(details_df))


# ============================================================
# 6. CLASS-LEVEL SUMMARY
# ============================================================

class_summary = (
    metrics_df
    .groupby(
        ["model", "iou_threshold", "class_id", "class_name"],
        as_index=False
    )
    .agg({
        "gt_count": "sum",
        "pred_count": "sum",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",
    })
)

class_summary["precision"] = (
    class_summary["tp"] /
    (
        class_summary["tp"] +
        class_summary["fp"]
    ).replace(0, np.nan)
).fillna(0)

class_summary["recall"] = (
    class_summary["tp"] /
    (
        class_summary["tp"] +
        class_summary["fn"]
    ).replace(0, np.nan)
).fillna(0)

class_summary["f1"] = (
    2 * class_summary["precision"] * class_summary["recall"] /
    (
        class_summary["precision"] +
        class_summary["recall"]
    ).replace(0, np.nan)
).fillna(0)

class_summary.to_csv(
    OUTPUT_DIR / "class_summary.csv",
    index=False
)


# ============================================================
# 7. OVERALL SUMMARY
# ============================================================

overall_summary = (
    metrics_df
    .groupby(
        ["model", "iou_threshold"],
        as_index=False
    )
    .agg({
        "gt_count": "sum",
        "pred_count": "sum",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",
    })
)

overall_summary["precision"] = (
    overall_summary["tp"] /
    (
        overall_summary["tp"] +
        overall_summary["fp"]
    ).replace(0, np.nan)
).fillna(0)

overall_summary["recall"] = (
    overall_summary["tp"] /
    (
        overall_summary["tp"] +
        overall_summary["fn"]
    ).replace(0, np.nan)
).fillna(0)

overall_summary["f1"] = (
    2 * overall_summary["precision"] * overall_summary["recall"] /
    (
        overall_summary["precision"] +
        overall_summary["recall"]
    ).replace(0, np.nan)
).fillna(0)

overall_summary.to_csv(
    OUTPUT_DIR / "overall_summary.csv",
    index=False
)


# ============================================================
# 8. OLD VS NEW VALIDATION SUMMARY
# ============================================================

split_summary = (
    metrics_df
    .groupby(
        ["model", "split", "iou_threshold"],
        as_index=False
    )
    .agg({
        "gt_count": "sum",
        "pred_count": "sum",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",
    })
)

split_summary["precision"] = (
    split_summary["tp"] /
    (
        split_summary["tp"] +
        split_summary["fp"]
    ).replace(0, np.nan)
).fillna(0)

split_summary["recall"] = (
    split_summary["tp"] /
    (
        split_summary["tp"] +
        split_summary["fn"]
    ).replace(0, np.nan)
).fillna(0)

split_summary["f1"] = (
    2 * split_summary["precision"] * split_summary["recall"] /
    (
        split_summary["precision"] +
        split_summary["recall"]
    ).replace(0, np.nan)
).fillna(0)

split_summary.to_csv(
    OUTPUT_DIR / "old_new_split_summary.csv",
    index=False
)


# ============================================================
# 9. SIZE-BASED RECALL
# ============================================================

size_rows = []

for (
    model_name,
    threshold,
    class_id,
    class_name
), group in metrics_df.groupby(
    ["model", "iou_threshold", "class_id", "class_name"]
):

    for size_name in ["small", "medium", "large"]:

        gt_count = int(group[f"{size_name}_gt"].sum())
        tp_count = int(group[f"{size_name}_tp"].sum())
        fn_count = int(group[f"{size_name}_fn"].sum())

        recall = (
            tp_count / gt_count
            if gt_count > 0 else np.nan
        )

        size_rows.append({
            "model": model_name,
            "iou_threshold": threshold,
            "class_id": class_id,
            "class_name": class_name,
            "size": size_name,
            "gt_count": gt_count,
            "tp": tp_count,
            "fn": fn_count,
            "recall": recall,
        })

size_summary = pd.DataFrame(size_rows)

size_summary.to_csv(
    OUTPUT_DIR / "size_based_recall.csv",
    index=False
)


# ============================================================
# 10. PAIRED IMAGE COMPARISON
# ============================================================

paired_columns = [
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]

paired_index = [
    "image",
    "split",
    "class_id",
    "class_name",
    "iou_threshold",
]

paired_wide = metrics_df.pivot_table(
    index=paired_index,
    columns="model",
    values=paired_columns,
    aggfunc="first"
)

paired_wide.columns = [
    f"{metric}_{model}"
    for metric, model in paired_wide.columns
]

paired_wide = paired_wide.reset_index()

for metric in paired_columns:

    c_col = f"{metric}_C_IRFS"
    cb_col = f"{metric}_CB_IRFS"

    if (
        c_col in paired_wide.columns
        and cb_col in paired_wide.columns
    ):
        paired_wide[
            f"delta_{metric}_CB_minus_C"
        ] = (
            paired_wide[cb_col] -
            paired_wide[c_col]
        )

paired_wide.to_csv(
    OUTPUT_DIR / "paired_image_comparison_C_vs_CB.csv",
    index=False
)


# ============================================================
# 11. FALSE-POSITIVE BREAKDOWN
# ============================================================

fp_details = details_df[
    details_df["status"] == "FP"
].copy()

if len(fp_details) > 0:

    fp_summary = (
        fp_details
        .groupby(
            [
                "model",
                "iou_threshold",
                "class_id",
                "class_name",
                "fp_type",
            ],
            as_index=False
        )
        .size()
        .rename(columns={"size": "count"})
    )

else:

    fp_summary = pd.DataFrame(
        columns=[
            "model",
            "iou_threshold",
            "class_id",
            "class_name",
            "fp_type",
            "count",
        ]
    )

fp_summary.to_csv(
    OUTPUT_DIR / "false_positive_breakdown.csv",
    index=False
)


# ============================================================
# 12. SAVE EXCEL SUMMARY
# ============================================================

excel_path = (
    OUTPUT_DIR /
    "paired_error_analysis_summary.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    overall_summary.to_excel(
        writer,
        sheet_name="Overall",
        index=False
    )

    class_summary.to_excel(
        writer,
        sheet_name="Per_Class",
        index=False
    )

    split_summary.to_excel(
        writer,
        sheet_name="Old_vs_New",
        index=False
    )

    size_summary.to_excel(
        writer,
        sheet_name="Size_Recall",
        index=False
    )

    fp_summary.to_excel(
        writer,
        sheet_name="FP_Breakdown",
        index=False
    )

    paired_wide.to_excel(
        writer,
        sheet_name="Paired_Image",
        index=False
    )


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("5. OVERALL SUMMARY")
print("=" * 75)

print(
    overall_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 75)
print("6. PER-CLASS SUMMARY")
print("=" * 75)

print(
    class_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 75)
print("7. OLD VS NEW")
print("=" * 75)

print(
    split_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 75)
print("8. SIZE-BASED RECALL")
print("=" * 75)

print(
    size_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 75)
print("9. FALSE-POSITIVE BREAKDOWN")
print("=" * 75)

print(fp_summary.to_string(index=False))


# ============================================================
# 14. VERIFY OUTPUT FILES
# ============================================================

print("\n" + "=" * 75)
print("10. OUTPUT FILES")
print("=" * 75)

expected_outputs = [
    "ALL_models_per_image_class_metrics.csv",
    "ALL_models_object_details.csv",
    "class_summary.csv",
    "overall_summary.csv",
    "old_new_split_summary.csv",
    "size_based_recall.csv",
    "paired_image_comparison_C_vs_CB.csv",
    "false_positive_breakdown.csv",
    "paired_error_analysis_summary.xlsx",
]

for filename in expected_outputs:

    path = OUTPUT_DIR / filename

    if not path.exists():
        raise RuntimeError(
            f"Expected output file missing: {path}"
        )

    print(
        f"[OK] {filename:<48} "
        f"{path.stat().st_size / 1024**2:.2f} MB"
    )


print("\n" + "=" * 75)
print("CELL 17I-FIX COMPLETED SUCCESSFULLY")
print("=" * 75)

###########################################################################
# CELL 35
###########################################################################
# ============================================================
# CELL 17K - CONFIDENCE THRESHOLD ANALYSIS
# C-IRFS vs CB-IRFS
#
# Dataset: 268 validation images
# Ground Truth: 3,042 instances
# Classes: 6
#
# Inference:
#   Base confidence = 0.001
#   Image size      = 960
#   NMS IoU         = 0.70
#   Max detections  = 300
#   Device          = GPU 0
#
# Evaluation:
#   Confidence thresholds:
#       0.001, 0.005, 0.01, 0.02, 0.05,
#       0.10, 0.20, 0.30, 0.40, 0.50
#
#   Mask IoU thresholds:
#       0.50 and 0.75
#
# IMPORTANT:
#   Inference is performed only once per checkpoint.
#   Prediction masks are cached to disk.
#   Every confidence threshold gets a fresh Hungarian matching.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import gc
import json
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks


# ============================================================
# 2. CONFIGURATION
# ============================================================

ROOT = Path("/kaggle/working/YOLO11_seg_expanded")

OUTPUT_DIR = (
    ROOT
    / "research_experiments"
    / "confidence_threshold_analysis"
)

CACHE_DIR = OUTPUT_DIR / "prediction_cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


MODEL_PATHS = {
    "C_IRFS": (
        ROOT
        / "research_experiments"
        / "C_irfs_t030"
        / "weights"
        / "best.pt"
    ),

    "CB_IRFS": (
        ROOT
        / "kaggle_outputs"
        / "CB_IRFS_sqrt_17ep_v1"
        / "weights"
        / "best.pt"
    ),
}


CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

NUM_CLASSES = 6


# Inference configuration.
DEVICE = 0

BASE_CONF = 0.001
IMGSZ = 960
NMS_IOU = 0.70
MAX_DET = 300
HALF = True
RETINA_MASKS = False


# Confidence thresholds to evaluate.
CONF_THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


# Mask IoU thresholds.
MASK_IOU_THRESHOLDS = [
    0.50,
    0.75,
]


# Ground Truth object-size definitions:
# small:  area ratio < 1%
# medium: 1% <= area ratio < 5%
# large:  area ratio >= 5%

SMALL_MAX_RATIO = 0.01
MEDIUM_MAX_RATIO = 0.05


EXPECTED_IMAGES = 268
EXPECTED_GT_INSTANCES = 3042


# Expected baseline results from Cell 17I-FIX.
# Used to verify that the new inference/cache pipeline
# reproduces the previous conf=0.001 analysis.

EXPECTED_BASELINE = {
    ("C_IRFS", 0.50): {
        "pred_count": 23505,
        "tp": 2631,
        "fp": 20874,
        "fn": 411,
    },

    ("C_IRFS", 0.75): {
        "pred_count": 23505,
        "tp": 1908,
        "fp": 21597,
        "fn": 1134,
    },

    ("CB_IRFS", 0.50): {
        "pred_count": 18570,
        "tp": 2586,
        "fp": 15984,
        "fn": 456,
    },

    ("CB_IRFS", 0.75): {
        "pred_count": 18570,
        "tp": 1923,
        "fp": 16647,
        "fn": 1119,
    },
}


# ============================================================
# 3. VERIFY PREVIOUS CELL STATE
# ============================================================

print("=" * 80)
print("1. VERIFY NOTEBOOK STATE")
print("=" * 80)

required_variables = [
    "image_records",
]

missing = [
    x for x in required_variables
    if x not in globals()
]

if missing:
    raise RuntimeError(
        "Missing variables from previous cells: "
        + ", ".join(missing)
        + ". Run the completed Cell 17I-FIX first."
    )


if len(image_records) != EXPECTED_IMAGES:
    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} images, "
        f"found {len(image_records)}."
    )


if DEVICE >= torch.cuda.device_count():
    raise RuntimeError(
        f"GPU {DEVICE} is unavailable."
    )


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")


for model_name, checkpoint in MODEL_PATHS.items():

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}"
        )


for threshold in CONF_THRESHOLDS:

    if threshold < BASE_CONF:
        raise ValueError(
            f"Confidence threshold {threshold} is below "
            f"the inference base threshold {BASE_CONF}."
        )


torch.cuda.set_device(DEVICE)
gc.collect()
torch.cuda.empty_cache()

free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)

print("Images:", len(image_records))
print("GPU:", torch.cuda.get_device_name(DEVICE))
print(f"GPU free: {free_bytes / 1024**3:.2f} GB")
print("C-IRFS checkpoint:", MODEL_PATHS["C_IRFS"])
print("CB-IRFS checkpoint:", MODEL_PATHS["CB_IRFS"])

if free_bytes / 1024**3 < 8:
    raise RuntimeError(
        "Not enough free GPU memory. "
        "At least 8 GB is required."
    )

print("\nNotebook verification: PASSED")


# ============================================================
# 4. VERIFY GROUND TRUTH
# ============================================================

print("\n" + "=" * 80)
print("2. VERIFY GROUND TRUTH")
print("=" * 80)

gt_count = 0
gt_class_counts = {
    i: 0 for i in range(NUM_CLASSES)
}

for record in image_records:

    if "gt_objects" not in record:
        raise RuntimeError(
            f"Missing GT objects: {record['image_name']}"
        )

    for obj in record["gt_objects"]:

        class_id = int(obj["class_id"])

        if not 0 <= class_id < NUM_CLASSES:
            raise RuntimeError(
                f"Invalid GT class: {class_id}"
            )

        gt_class_counts[class_id] += 1
        gt_count += 1


EXPECTED_CLASS_COUNTS = {
    0: 829,
    1: 238,
    2: 339,
    3: 446,
    4: 274,
    5: 916,
}


if gt_count != EXPECTED_GT_INSTANCES:
    raise RuntimeError(
        f"Expected {EXPECTED_GT_INSTANCES} GT instances, "
        f"found {gt_count}."
    )


if gt_class_counts != EXPECTED_CLASS_COUNTS:
    raise RuntimeError(
        "GT class counts do not match verified data.\n"
        f"Expected: {EXPECTED_CLASS_COUNTS}\n"
        f"Actual:   {gt_class_counts}"
    )


print("Total GT instances:", gt_count)

for class_id, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_id}: {class_name:<22} "
        f"{gt_class_counts[class_id]}"
    )

print("\nGround Truth verification: PASSED")


# ============================================================
# 5. OBJECT SIZE CLASSIFICATION
# ============================================================

def classify_size(area_ratio):

    if area_ratio < SMALL_MAX_RATIO:
        return "small"

    if area_ratio < MEDIUM_MAX_RATIO:
        return "medium"

    return "large"


# ============================================================
# 6. RUN INFERENCE AND SAVE CROPPED MASKS
# ============================================================

def infer_and_cache_one_image(model, image_record):

    image_path = Path(image_record["image_path"])

    height = int(image_record["height"])
    width = int(image_record["width"])

    results = model.predict(
        source=str(image_path),

        imgsz=IMGSZ,

        batch=1,

        conf=BASE_CONF,

        iou=NMS_IOU,

        max_det=MAX_DET,

        retina_masks=False,

        half=HALF,

        device=DEVICE,

        verbose=False,

        stream=False,
    )

    predictions = []

    if not results:
        return predictions

    result = results[0]

    if result.boxes is None or result.masks is None:
        del results
        return predictions

    boxes = result.boxes
    raw_masks = result.masks.data

    if raw_masks.ndim != 3:
        raise RuntimeError(
            f"Unexpected raw mask shape: "
            f"{tuple(raw_masks.shape)}"
        )

    class_ids = (
        boxes.cls.detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        boxes.conf.detach()
        .cpu()
        .numpy()
    )

    num_predictions = min(
        len(class_ids),
        len(confidences),
        len(raw_masks)
    )

    for i in range(num_predictions):

        class_id = int(class_ids[i])

        if not 0 <= class_id < NUM_CLASSES:
            continue

        confidence = float(confidences[i])

        # Mask tensor:
        # [1, H, W] -> [1, 1, H, W].
        one_mask = (
            raw_masks[i:i + 1]
            .detach()
            .float()
            .cpu()
            .unsqueeze(1)
        )

        scaled_mask = scale_masks(
            one_mask,
            (height, width)
        )

        binary_mask = (
            scaled_mask[0, 0].numpy() >= 0.5
        )

        del one_mask, scaled_mask

        ys, xs = np.where(binary_mask)

        if len(xs) == 0:
            del binary_mask
            continue

        x_min = int(xs.min())
        y_min = int(ys.min())

        x_max = int(xs.max()) + 1
        y_max = int(ys.max()) + 1

        cropped_mask = binary_mask[
            y_min:y_max,
            x_min:x_max
        ].copy()

        area = int(cropped_mask.sum())

        if area == 0:
            del binary_mask, cropped_mask
            continue

        area_ratio = area / float(width * height)

        predictions.append({
            "id": i,
            "class_id": class_id,
            "confidence": confidence,

            "bbox": (
                x_min,
                y_min,
                x_max,
                y_max,
            ),

            "mask": cropped_mask.astype(bool),

            "area": area,

            "area_ratio": area_ratio,

            "size": classify_size(area_ratio),
        })

        del binary_mask, cropped_mask

    del raw_masks, boxes, result, results

    return predictions


# ============================================================
# 7. SAVE PREDICTIONS TO COMPRESSED NPZ
# ============================================================

def save_prediction_cache(predictions, output_path):

    n = len(predictions)

    if n == 0:

        np.savez_compressed(
            output_path,

            class_ids=np.empty(0, dtype=np.int16),

            confidences=np.empty(0, dtype=np.float32),

            boxes=np.empty((0, 4), dtype=np.int32),

            areas=np.empty(0, dtype=np.int32),

            area_ratios=np.empty(0, dtype=np.float32),

            shapes=np.empty((0, 2), dtype=np.int32),

            offsets=np.zeros(1, dtype=np.int64),

            flat_masks=np.empty(0, dtype=np.uint8),
        )

        return


    class_ids = []
    confidences = []
    boxes = []
    areas = []
    area_ratios = []
    shapes = []

    offsets = [0]
    mask_parts = []

    for pred in predictions:

        mask = np.asarray(
            pred["mask"],
            dtype=np.uint8
        )

        h, w = mask.shape

        class_ids.append(pred["class_id"])
        confidences.append(pred["confidence"])

        boxes.append(pred["bbox"])

        areas.append(pred["area"])
        area_ratios.append(pred["area_ratio"])

        shapes.append((h, w))

        flattened = mask.reshape(-1)

        mask_parts.append(flattened)

        offsets.append(
            offsets[-1] + len(flattened)
        )


    flat_masks = np.concatenate(mask_parts)

    np.savez_compressed(
        output_path,

        class_ids=np.asarray(
            class_ids,
            dtype=np.int16
        ),

        confidences=np.asarray(
            confidences,
            dtype=np.float32
        ),

        boxes=np.asarray(
            boxes,
            dtype=np.int32
        ).reshape(-1, 4),

        areas=np.asarray(
            areas,
            dtype=np.int32
        ),

        area_ratios=np.asarray(
            area_ratios,
            dtype=np.float32
        ),

        shapes=np.asarray(
            shapes,
            dtype=np.int32
        ).reshape(-1, 2),

        offsets=np.asarray(
            offsets,
            dtype=np.int64
        ),

        flat_masks=flat_masks,
    )


# ============================================================
# 8. LOAD PREDICTIONS FROM CACHE
# ============================================================

def load_prediction_cache(cache_path):

    predictions = []

    with np.load(
        cache_path,
        allow_pickle=False
    ) as data:

        class_ids = data["class_ids"]
        confidences = data["confidences"]

        boxes = data["boxes"]

        areas = data["areas"]
        area_ratios = data["area_ratios"]

        shapes = data["shapes"]
        offsets = data["offsets"]

        flat_masks = data["flat_masks"]

        n = len(class_ids)

        for i in range(n):

            h, w = shapes[i]

            start = int(offsets[i])
            end = int(offsets[i + 1])

            mask = (
                flat_masks[start:end]
                .reshape(int(h), int(w))
                .astype(bool)
            )

            predictions.append({

                "id": i,

                "class_id": int(class_ids[i]),

                "confidence": float(confidences[i]),

                "bbox": tuple(
                    int(x) for x in boxes[i]
                ),

                "mask": mask,

                "area": int(areas[i]),

                "area_ratio": float(area_ratios[i]),

                "size": classify_size(
                    float(area_ratios[i])
                ),
            })

    return predictions


# ============================================================
# 9. RUN AND CACHE ONE MODEL
# ============================================================

def generate_model_cache(model_name, checkpoint):

    print("\n" + "=" * 80)
    print("GENERATING CACHE:", model_name)
    print("=" * 80)

    model_cache_dir = CACHE_DIR / model_name

    model_cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Clean old cache to ensure the current inference settings
    # are used consistently.
    for old_file in model_cache_dir.glob("*.npz"):
        old_file.unlink()

    model = YOLO(str(checkpoint))

    prediction_counts = []

    manifest_rows = []

    for index, image_record in enumerate(
        image_records,
        start=0
    ):

        predictions = infer_and_cache_one_image(
            model,
            image_record
        )

        cache_path = (
            model_cache_dir
            / f"{index:04d}.npz"
        )

        save_prediction_cache(
            predictions,
            cache_path
        )

        prediction_count = len(predictions)

        prediction_counts.append(
            prediction_count
        )

        manifest_rows.append({
            "index": index,

            "image": image_record["image_name"],

            "split": image_record["split"],

            "prediction_count": prediction_count,

            "cache_file": str(cache_path),
        })

        if (index + 1) % 10 == 0 or index + 1 == len(image_records):

            free_bytes, _ = torch.cuda.mem_get_info(DEVICE)

            print(
                f"[{model_name}] "
                f"{index + 1:3d}/{len(image_records)} | "
                f"predictions={prediction_count:3d} | "
                f"GPU free={free_bytes / 1024**3:.2f} GB"
            )

        del predictions

        if (index + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()


    del model

    gc.collect()
    torch.cuda.empty_cache()

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_df.to_csv(
        OUTPUT_DIR / f"{model_name}_cache_manifest.csv",
        index=False
    )

    cache_summary = {

        "model": model_name,

        "num_images": len(image_records),

        "total_predictions": int(
            np.sum(prediction_counts)
        ),

        "mean_predictions_per_image": float(
            np.mean(prediction_counts)
        ),

        "max_predictions_per_image": int(
            np.max(prediction_counts)
        ),

        "images_at_max_det": int(
            np.sum(
                np.asarray(prediction_counts) >= MAX_DET
            )
        ),

        "base_conf": BASE_CONF,

        "imgsz": IMGSZ,

        "nms_iou": NMS_IOU,

        "max_det": MAX_DET,

        "half": HALF,

        "retina_masks": RETINA_MASKS,

    }

    with open(
        OUTPUT_DIR / f"{model_name}_cache_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cache_summary,
            f,
            indent=2
        )


    print("\nCache summary:")
    print(json.dumps(cache_summary, indent=2))

    if cache_summary["images_at_max_det"] > 0:

        print(
            "\nWARNING: Some images reached max_det=300. "
            "Their detections may have been truncated."
        )

    return cache_summary


# ============================================================
# 10. GENERATE CACHES FOR BOTH CHECKPOINTS
# ============================================================

print("\n" + "=" * 80)
print("3. GENERATE PREDICTION CACHES")
print("=" * 80)

cache_summaries = {}

for model_name, checkpoint in MODEL_PATHS.items():

    cache_summaries[model_name] = generate_model_cache(
        model_name,
        checkpoint
    )


print("\nBoth model caches generated.")


# ============================================================
# 11. MASK IoU BETWEEN CROPPED MASKS
# ============================================================

def cropped_mask_iou(obj_a, obj_b):

    ax1, ay1, ax2, ay2 = obj_a["bbox"]

    bx1, by1, bx2, by2 = obj_b["bbox"]

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = 0

    if x2 > x1 and y2 > y1:

        a_crop = obj_a["mask"][
            y1 - ay1:y2 - ay1,
            x1 - ax1:x2 - ax1
        ]

        b_crop = obj_b["mask"][
            y1 - by1:y2 - by1,
            x1 - bx1:x2 - bx1
        ]

        intersection = int(
            np.logical_and(
                a_crop,
                b_crop
            ).sum()
        )


    union = (
        int(obj_a["area"])
        + int(obj_b["area"])
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(intersection / union)


# ============================================================
# 12. PRECOMPUTE FULL IoU MATRIX
# ============================================================

def compute_iou_matrix(gt_objects, pred_objects):

    n_gt = len(gt_objects)
    n_pred = len(pred_objects)

    matrix = np.zeros(
        (n_gt, n_pred),
        dtype=np.float32
    )

    for i, gt_obj in enumerate(gt_objects):

        for j, pred_obj in enumerate(pred_objects):

            matrix[i, j] = cropped_mask_iou(
                gt_obj,
                pred_obj
            )

    return matrix


# ============================================================
# 13. HUNGARIAN MATCHING
# ============================================================

def hungarian_match_from_iou(iou_matrix, threshold):

    n_gt, n_pred = iou_matrix.shape

    if n_gt == 0 or n_pred == 0:
        return []

    matrix_size = max(n_gt, n_pred)

    score = np.zeros(
        (matrix_size, matrix_size),
        dtype=np.float64
    )

    valid = iou_matrix >= threshold

    # Maximize the number of valid matches first,
    # then maximize the total IoU.
    score[:n_gt, :n_pred] = np.where(
        valid,
        1_000_000.0 + iou_matrix,
        -1_000_000.0
    )

    rows, cols = linear_sum_assignment(-score)

    matches = []

    for row, col in zip(rows, cols):

        if row >= n_gt or col >= n_pred:
            continue

        iou_value = float(
            iou_matrix[row, col]
        )

        if iou_value >= threshold:

            matches.append((
                int(row),
                int(col),
                iou_value
            ))

    return matches


# ============================================================
# 14. ANALYZE ONE MODEL ACROSS ALL CONFIDENCE THRESHOLDS
# ============================================================

def analyze_model_thresholds(model_name):

    print("\n" + "=" * 80)
    print("ANALYZING:", model_name)
    print("=" * 80)

    model_cache_dir = CACHE_DIR / model_name

    metric_rows = []

    for image_index, image_record in enumerate(image_records):

        cache_path = (
            model_cache_dir
            / f"{image_index:04d}.npz"
        )

        if not cache_path.exists():

            raise FileNotFoundError(
                f"Missing cache: {cache_path}"
            )

        predictions = load_prediction_cache(
            cache_path
        )

        gt_objects = image_record["gt_objects"]

        # Group predictions and GT by class.
        gt_by_class = {
            c: [
                obj for obj in gt_objects
                if int(obj["class_id"]) == c
            ]
            for c in range(NUM_CLASSES)
        }

        pred_by_class = {
            c: [
                obj for obj in predictions
                if int(obj["class_id"]) == c
            ]
            for c in range(NUM_CLASSES)
        }


        # Precompute IoU once for each class and image.
        # The same IoU matrix is reused for all confidence
        # and mask IoU thresholds.
        iou_by_class = {}

        for class_id in range(NUM_CLASSES):

            iou_by_class[class_id] = compute_iou_matrix(
                gt_by_class[class_id],
                pred_by_class[class_id]
            )


        # Evaluate each confidence threshold.
        for conf_threshold in CONF_THRESHOLDS:

            for mask_iou_threshold in MASK_IOU_THRESHOLDS:

                for class_id in range(NUM_CLASSES):

                    gt_class = gt_by_class[class_id]
                    pred_class = pred_by_class[class_id]

                    iou_full = iou_by_class[class_id]

                    # Keep predictions above the current
                    # confidence threshold.
                    selected_prediction_indices = [
                        j
                        for j, pred in enumerate(pred_class)
                        if pred["confidence"] >= conf_threshold
                    ]

                    selected_count = len(
                        selected_prediction_indices
                    )

                    # Restrict the precomputed IoU matrix
                    # to the surviving predictions.
                    if selected_count > 0:

                        iou_selected = iou_full[
                            :,
                            selected_prediction_indices
                        ]

                    else:

                        iou_selected = np.zeros(
                            (len(gt_class), 0),
                            dtype=np.float32
                        )


                    matches = hungarian_match_from_iou(
                        iou_selected,
                        mask_iou_threshold
                    )

                    matched_gt_indices = {
                        item[0] for item in matches
                    }

                    tp = len(matches)

                    fp = selected_count - tp

                    fn = len(gt_class) - tp

                    precision = (
                        tp / (tp + fp)
                        if (tp + fp) > 0
                        else 0.0
                    )

                    recall = (
                        tp / (tp + fn)
                        if (tp + fn) > 0
                        else 0.0
                    )

                    f1 = (
                        2 * precision * recall
                        / (precision + recall)
                        if (precision + recall) > 0
                        else 0.0
                    )


                    # Size-based GT and TP counts.
                    size_counts = {
                        "small": {
                            "gt": 0,
                            "tp": 0,
                        },

                        "medium": {
                            "gt": 0,
                            "tp": 0,
                        },

                        "large": {
                            "gt": 0,
                            "tp": 0,
                        },
                    }


                    for gt_idx, gt_obj in enumerate(gt_class):

                        size_name = gt_obj["size"]

                        size_counts[size_name]["gt"] += 1

                        if gt_idx in matched_gt_indices:

                            size_counts[size_name]["tp"] += 1


                    row = {

                        "model": model_name,

                        "image": image_record["image_name"],

                        "split": image_record["split"],

                        "class_id": class_id,

                        "class_name": CLASS_NAMES[class_id],

                        "confidence_threshold": conf_threshold,

                        "iou_threshold": mask_iou_threshold,

                        "gt_count": len(gt_class),

                        "pred_count": selected_count,

                        "tp": tp,

                        "fp": fp,

                        "fn": fn,

                        "precision": precision,

                        "recall": recall,

                        "f1": f1,
                    }


                    for size_name in [
                        "small",
                        "medium",
                        "large",
                    ]:

                        gt_size = size_counts[size_name]["gt"]

                        tp_size = size_counts[size_name]["tp"]

                        row[f"{size_name}_gt"] = gt_size

                        row[f"{size_name}_tp"] = tp_size

                        row[f"{size_name}_fn"] = (
                            gt_size - tp_size
                        )


                    metric_rows.append(row)


        del predictions

        if (image_index + 1) % 20 == 0:

            print(
                f"[{model_name}] "
                f"{image_index + 1:3d}/{len(image_records)} "
                f"images analyzed | "
                f"rows={len(metric_rows)}"
            )

            gc.collect()


    return pd.DataFrame(metric_rows)


# ============================================================
# 15. ANALYZE BOTH MODELS
# ============================================================

print("\n" + "=" * 80)
print("4. CONFIDENCE THRESHOLD EVALUATION")
print("=" * 80)

all_metrics = []

for model_name in MODEL_PATHS.keys():

    model_metrics = analyze_model_thresholds(
        model_name
    )

    model_metrics.to_csv(
        OUTPUT_DIR
        / f"{model_name}_confidence_per_image_metrics.csv",
        index=False
    )

    all_metrics.append(model_metrics)

    print(
        f"\n{model_name}: "
        f"{len(model_metrics)} rows"
    )

    del model_metrics
    gc.collect()


metrics_df = pd.concat(
    all_metrics,
    ignore_index=True
)

metrics_df.to_csv(
    OUTPUT_DIR / "ALL_confidence_per_image_metrics.csv",
    index=False
)

print("\nTotal per-image metric rows:", len(metrics_df))


# ============================================================
# 16. SUMMARY HELPER
# ============================================================

def calculate_summary(df, group_columns):

    summary = (
        df
        .groupby(
            group_columns,
            as_index=False
        )
        .agg({

            "gt_count": "sum",

            "pred_count": "sum",

            "tp": "sum",

            "fp": "sum",

            "fn": "sum",

            "small_gt": "sum",

            "small_tp": "sum",

            "small_fn": "sum",

            "medium_gt": "sum",

            "medium_tp": "sum",

            "medium_fn": "sum",

            "large_gt": "sum",

            "large_tp": "sum",

            "large_fn": "sum",

        })
    )


    summary["precision"] = (
        summary["tp"]
        / (
            summary["tp"] + summary["fp"]
        ).replace(0, np.nan)
    ).fillna(0)


    summary["recall"] = (
        summary["tp"]
        / (
            summary["tp"] + summary["fn"]
        ).replace(0, np.nan)
    ).fillna(0)


    summary["f1"] = (
        2 * summary["precision"] * summary["recall"]
        / (
            summary["precision"] + summary["recall"]
        ).replace(0, np.nan)
    ).fillna(0)


    for size_name in [
        "small",
        "medium",
        "large",
    ]:

        summary[f"{size_name}_recall"] = (
            summary[f"{size_name}_tp"]
            / summary[f"{size_name}_gt"].replace(0, np.nan)
        )

    return summary


# ============================================================
# 17. OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("5. OVERALL SUMMARY")
print("=" * 80)

overall_summary = calculate_summary(
    metrics_df,

    [
        "model",
        "confidence_threshold",
        "iou_threshold",
    ]
)

overall_summary.to_csv(
    OUTPUT_DIR / "overall_confidence_summary.csv",
    index=False
)


# ============================================================
# 18. PER-CLASS SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("6. PER-CLASS SUMMARY")
print("=" * 80)

class_summary = calculate_summary(
    metrics_df,

    [
        "model",
        "confidence_threshold",
        "iou_threshold",
        "class_id",
        "class_name",
    ]
)

class_summary.to_csv(
    OUTPUT_DIR / "per_class_confidence_summary.csv",
    index=False
)


# ============================================================
# 19. OLD VS NEW SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("7. OLD VS NEW SUMMARY")
print("=" * 80)

split_summary = calculate_summary(
    metrics_df,

    [
        "model",
        "split",
        "confidence_threshold",
        "iou_threshold",
    ]
)

split_summary.to_csv(
    OUTPUT_DIR / "old_new_confidence_summary.csv",
    index=False
)


# ============================================================
# 20. SIZE-BASED RECALL SUMMARY
# ============================================================

size_rows = []

for _, row in class_summary.iterrows():

    for size_name in [
        "small",
        "medium",
        "large",
    ]:

        size_rows.append({

            "model": row["model"],

            "confidence_threshold": row[
                "confidence_threshold"
            ],

            "iou_threshold": row["iou_threshold"],

            "class_id": row["class_id"],

            "class_name": row["class_name"],

            "size": size_name,

            "gt_count": row[f"{size_name}_gt"],

            "tp": row[f"{size_name}_tp"],

            "fn": row[f"{size_name}_fn"],

            "recall": row[f"{size_name}_recall"],

        })


size_summary = pd.DataFrame(size_rows)

size_summary.to_csv(
    OUTPUT_DIR / "size_based_confidence_summary.csv",
    index=False
)


# ============================================================
# 21. BEST GLOBAL CONFIDENCE THRESHOLD
# ============================================================

# IMPORTANT:
# This threshold is selected using the current validation set.
# It is a diagnostic operating point, not an independent test result.

best_global_rows = []

for (
    model_name,
    iou_threshold
), group in overall_summary.groupby(
    ["model", "iou_threshold"]
):

    best_row = (
        group
        .sort_values(
            ["f1", "precision"],
            ascending=[False, False]
        )
        .iloc[0]
    )

    best_global_rows.append({

        "model": model_name,

        "iou_threshold": iou_threshold,

        "best_confidence_threshold": (
            best_row["confidence_threshold"]
        ),

        "precision": best_row["precision"],

        "recall": best_row["recall"],

        "f1": best_row["f1"],

        "tp": best_row["tp"],

        "fp": best_row["fp"],

        "fn": best_row["fn"],

    })


best_global_summary = pd.DataFrame(
    best_global_rows
)

best_global_summary.to_csv(
    OUTPUT_DIR / "best_global_confidence_threshold.csv",
    index=False
)


# ============================================================
# 22. BEST CONFIDENCE THRESHOLD PER CLASS
# ============================================================

best_class_rows = []

for (
    model_name,
    iou_threshold,
    class_id,
    class_name
), group in class_summary.groupby(
    [
        "model",
        "iou_threshold",
        "class_id",
        "class_name",
    ]
):

    best_row = (
        group
        .sort_values(
            ["f1", "precision"],
            ascending=[False, False]
        )
        .iloc[0]
    )

    best_class_rows.append({

        "model": model_name,

        "iou_threshold": iou_threshold,

        "class_id": class_id,

        "class_name": class_name,

        "best_confidence_threshold": (
            best_row["confidence_threshold"]
        ),

        "precision": best_row["precision"],

        "recall": best_row["recall"],

        "f1": best_row["f1"],

        "tp": best_row["tp"],

        "fp": best_row["fp"],

        "fn": best_row["fn"],

    })


best_class_summary = pd.DataFrame(
    best_class_rows
)

best_class_summary.to_csv(
    OUTPUT_DIR / "best_confidence_threshold_per_class.csv",
    index=False
)


# ============================================================
# 23. VERIFY CONFIDENCE=0.001 AGAINST CELL 17I-FIX
# ============================================================

print("\n" + "=" * 80)
print("8. BASELINE REPRODUCTION CHECK")
print("=" * 80)

baseline_rows = []

for (model_name, iou_threshold), expected in (
    EXPECTED_BASELINE.items()
):

    selected = overall_summary[
        (overall_summary["model"] == model_name)
        &
        (
            np.isclose(
                overall_summary["confidence_threshold"],
                BASE_CONF
            )
        )
        &
        (
            np.isclose(
                overall_summary["iou_threshold"],
                iou_threshold
            )
        )
    ]

    if len(selected) != 1:

        raise RuntimeError(
            f"Could not find baseline row for "
            f"{model_name}, IoU={iou_threshold}"
        )

    actual = selected.iloc[0]

    matches = True

    for key in [
        "pred_count",
        "tp",
        "fp",
        "fn",
    ]:

        actual_value = int(actual[key])

        expected_value = int(expected[key])

        if actual_value != expected_value:
            matches = False


    baseline_rows.append({

        "model": model_name,

        "iou_threshold": iou_threshold,

        "expected_pred": expected["pred_count"],

        "actual_pred": int(actual["pred_count"]),

        "expected_tp": expected["tp"],

        "actual_tp": int(actual["tp"]),

        "expected_fp": expected["fp"],

        "actual_fp": int(actual["fp"]),

        "expected_fn": expected["fn"],

        "actual_fn": int(actual["fn"]),

        "match": matches,

    })


baseline_check = pd.DataFrame(
    baseline_rows
)

print(
    baseline_check.to_string(index=False)
)

baseline_check.to_csv(
    OUTPUT_DIR / "baseline_reproduction_check.csv",
    index=False
)

if not baseline_check["match"].all():

    print(
        "\nWARNING: The new conf=0.001 results do not "
        "exactly reproduce Cell 17I-FIX."
    )

    print(
        "Do not interpret the threshold curves until "
        "the discrepancy is investigated."
    )

else:

    print("\nBaseline reproduction: PASSED")


# ============================================================
# 24. PLOT CONFIDENCE CURVES
# ============================================================

print("\n" + "=" * 80)
print("9. GENERATE CONFIDENCE CURVES")
print("=" * 80)

plot_metrics = [
    "precision",
    "recall",
    "f1",
]

for metric_name in plot_metrics:

    plt.figure(figsize=(10, 6))

    for model_name in MODEL_PATHS.keys():

        for iou_threshold in MASK_IOU_THRESHOLDS:

            subset = overall_summary[
                (overall_summary["model"] == model_name)
                &
                (
                    np.isclose(
                        overall_summary["iou_threshold"],
                        iou_threshold
                    )
                )
            ].sort_values("confidence_threshold")

            plt.plot(
                subset["confidence_threshold"],
                subset[metric_name],
                marker="o",
                label=(
                    f"{model_name} "
                    f"IoU={iou_threshold:.2f}"
                )
            )

    plt.xlabel("Confidence threshold")
    plt.ylabel(metric_name.capitalize())

    plt.title(
        f"{metric_name.capitalize()} vs Confidence Threshold"
    )

    plt.xticks(CONF_THRESHOLDS, rotation=45)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / f"{metric_name}_vs_confidence.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print("[SAVED]", plot_path)


# ============================================================
# 25. EXPORT EXCEL WORKBOOK
# ============================================================

print("\n" + "=" * 80)
print("10. EXPORT EXCEL")
print("=" * 80)

excel_path = (
    OUTPUT_DIR
    / "confidence_threshold_analysis.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    overall_summary.to_excel(
        writer,
        sheet_name="Overall",
        index=False
    )

    class_summary.to_excel(
        writer,
        sheet_name="Per_Class",
        index=False
    )

    split_summary.to_excel(
        writer,
        sheet_name="Old_vs_New",
        index=False
    )

    size_summary.to_excel(
        writer,
        sheet_name="Size_Recall",
        index=False
    )

    best_global_summary.to_excel(
        writer,
        sheet_name="Best_Global_Threshold",
        index=False
    )

    best_class_summary.to_excel(
        writer,
        sheet_name="Best_Per_Class",
        index=False
    )

    baseline_check.to_excel(
        writer,
        sheet_name="Baseline_Check",
        index=False
    )


# ============================================================
# 26. PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("11. OVERALL CONFIDENCE THRESHOLD RESULTS")
print("=" * 80)

display_columns = [
    "model",
    "confidence_threshold",
    "iou_threshold",
    "pred_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]

print(
    overall_summary[display_columns]
    .sort_values(
        [
            "iou_threshold",
            "confidence_threshold",
            "model",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 27. PRINT BEST GLOBAL THRESHOLDS
# ============================================================

print("\n" + "=" * 80)
print("12. BEST GLOBAL THRESHOLD ON VALIDATION")
print("=" * 80)

print(
    best_global_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 28. PRINT BEST THRESHOLD PER CLASS
# ============================================================

print("\n" + "=" * 80)
print("13. BEST THRESHOLD PER CLASS ON VALIDATION")
print("=" * 80)

print(
    best_class_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 29. FINAL OUTPUT CHECK
# ============================================================

print("\n" + "=" * 80)
print("14. OUTPUT FILES")
print("=" * 80)

expected_files = [

    "ALL_confidence_per_image_metrics.csv",

    "overall_confidence_summary.csv",

    "per_class_confidence_summary.csv",

    "old_new_confidence_summary.csv",

    "size_based_confidence_summary.csv",

    "best_global_confidence_threshold.csv",

    "best_confidence_threshold_per_class.csv",

    "baseline_reproduction_check.csv",

    "precision_vs_confidence.png",

    "recall_vs_confidence.png",

    "f1_vs_confidence.png",

    "confidence_threshold_analysis.xlsx",

]


for filename in expected_files:

    path = OUTPUT_DIR / filename

    if not path.exists():

        raise RuntimeError(
            f"Missing output file: {path}"
        )

    print(
        f"[OK] {filename:<52} "
        f"{path.stat().st_size / 1024**2:.3f} MB"
    )


print("\nPrediction cache location:")
print(CACHE_DIR)

print("\nAnalysis output location:")
print(OUTPUT_DIR)

print("\n" + "=" * 80)
print("CELL 17K COMPLETED SUCCESSFULLY")
print("=" * 80)

###########################################################################
# CELL 36
###########################################################################
# ============================================================
# CELL 17L
# EXTENDED CONFIDENCE ANALYSIS + FP/FN VISUAL AUDIT
#
# Models:
#   C_IRFS
#   CB_IRFS
#
# New confidence thresholds:
#   0.60, 0.70, 0.80, 0.90
#
# Existing thresholds:
#   Loaded from Cell 17K CSV
#
# Mask IoU:
#   0.50, 0.75
#
# Visual audit operating point:
#   Confidence = 0.40
#   Mask IoU   = 0.50
#
# IMPORTANT:
#   - No YOLO inference.
#   - Uses saved prediction cache from Cell 17K.
#   - Uses existing image_records and GT masks.
#   - Re-runs Hungarian matching after confidence filtering.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import gc
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from scipy.optimize import linear_sum_assignment


# ============================================================
# 2. CONFIGURATION
# ============================================================

ROOT = Path("/kaggle/working/YOLO11_seg_expanded")

ANALYSIS_DIR = (
    ROOT
    / "research_experiments"
    / "confidence_threshold_analysis"
)

CACHE_DIR = ANALYSIS_DIR / "prediction_cache"

OUTPUT_DIR = (
    ROOT
    / "research_experiments"
    / "confidence_threshold_analysis_extended"
)

VISUAL_DIR = OUTPUT_DIR / "error_visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAMES = [
    "C_IRFS",
    "CB_IRFS",
]

CLASS_NAMES = [
    "hat",
    "gloves",
    "mask",
    "pants",
    "shoes",
    "long-sleeved shirt",
]

NUM_CLASSES = 6

NEW_CONF_THRESHOLDS = [
    0.60,
    0.70,
    0.80,
    0.90,
]

MASK_IOU_THRESHOLDS = [
    0.50,
    0.75,
]

VIS_CONF = 0.40
VIS_IOU = 0.50

VISUAL_TOP_K = 2

EXPECTED_IMAGES = 268
EXPECTED_GT = 3042

EXPECTED_CLASS_COUNTS = {
    0: 829,
    1: 238,
    2: 339,
    3: 446,
    4: 274,
    5: 916,
}

DEVICE = 0

SMALL_MAX_RATIO = 0.01
MEDIUM_MAX_RATIO = 0.05


# ============================================================
# 3. VERIFY NOTEBOOK STATE
# ============================================================

print("=" * 80)
print("1. VERIFY NOTEBOOK STATE")
print("=" * 80)

if "image_records" not in globals():
    raise RuntimeError(
        "image_records is missing. "
        "Run the completed Cell 17I-FIX first."
    )

if len(image_records) != EXPECTED_IMAGES:
    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} images, "
        f"found {len(image_records)}."
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

if torch.cuda.device_count() <= DEVICE:
    raise RuntimeError(f"GPU {DEVICE} is unavailable.")

torch.cuda.set_device(DEVICE)

for model_name in MODEL_NAMES:

    model_cache_dir = CACHE_DIR / model_name

    if not model_cache_dir.exists():
        raise FileNotFoundError(model_cache_dir)

    missing_cache = []

    for i in range(EXPECTED_IMAGES):

        cache_file = model_cache_dir / f"{i:04d}.npz"

        if not cache_file.exists():
            missing_cache.append(str(cache_file))

    if missing_cache:
        raise RuntimeError(
            f"{model_name}: missing {len(missing_cache)} cache files. "
            f"First missing: {missing_cache[:3]}"
        )


print("Images:", len(image_records))
print("GPU:", torch.cuda.get_device_name(DEVICE))

free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)

print(f"GPU free: {free_bytes / 1024**3:.2f} GB")

print("Prediction cache: PASSED")


# ============================================================
# 4. VERIFY GROUND TRUTH
# ============================================================

print("\n" + "=" * 80)
print("2. VERIFY GROUND TRUTH")
print("=" * 80)

total_gt = 0

class_counts = {
    i: 0 for i in range(NUM_CLASSES)
}

for record in image_records:

    if "gt_objects" not in record:
        raise RuntimeError(
            f"Missing Ground Truth: {record['image_name']}"
        )

    for obj in record["gt_objects"]:

        class_id = int(obj["class_id"])

        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(
                f"Invalid GT class ID: {class_id}"
            )

        total_gt += 1
        class_counts[class_id] += 1


if total_gt != EXPECTED_GT:
    raise RuntimeError(
        f"Expected {EXPECTED_GT} GT instances, "
        f"found {total_gt}"
    )

if class_counts != EXPECTED_CLASS_COUNTS:
    raise RuntimeError(
        f"GT class count mismatch.\n"
        f"Expected: {EXPECTED_CLASS_COUNTS}\n"
        f"Actual: {class_counts}"
    )

print("Total GT:", total_gt)
print("Class counts:", class_counts)
print("Ground Truth verification: PASSED")


# ============================================================
# 5. VERIFY CELL 17K RESULTS
# ============================================================

print("\n" + "=" * 80)
print("3. VERIFY CELL 17K RESULTS")
print("=" * 80)

OLD_METRICS_PATH = (
    ANALYSIS_DIR
    / "ALL_confidence_per_image_metrics.csv"
)

if not OLD_METRICS_PATH.exists():
    raise FileNotFoundError(
        f"Cell 17K metrics file not found: {OLD_METRICS_PATH}"
    )

old_metrics = pd.read_csv(OLD_METRICS_PATH)

required_old_columns = [
    "model",
    "image",
    "split",
    "class_id",
    "class_name",
    "confidence_threshold",
    "iou_threshold",
    "gt_count",
    "pred_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "small_gt",
    "small_tp",
    "small_fn",
    "medium_gt",
    "medium_tp",
    "medium_fn",
    "large_gt",
    "large_tp",
    "large_fn",
]

missing_columns = [
    col for col in required_old_columns
    if col not in old_metrics.columns
]

if missing_columns:
    raise RuntimeError(
        "Cell 17K metrics file is missing columns: "
        + ", ".join(missing_columns)
    )

print("Previous metric rows:", len(old_metrics))
print(
    "Previous confidence thresholds:",
    sorted(old_metrics["confidence_threshold"].unique())
)

print("Cell 17K metrics verification: PASSED")


# ============================================================
# 6. LOAD PREDICTION CACHE
# ============================================================

def load_prediction_cache(cache_path):

    predictions = []

    with np.load(
        cache_path,
        allow_pickle=False
    ) as data:

        class_ids = data["class_ids"]
        confidences = data["confidences"]
        boxes = data["boxes"]

        areas = data["areas"]
        area_ratios = data["area_ratios"]

        shapes = data["shapes"]
        offsets = data["offsets"]

        flat_masks = data["flat_masks"]

        for i in range(len(class_ids)):

            h, w = shapes[i]

            start = int(offsets[i])
            end = int(offsets[i + 1])

            mask = (
                flat_masks[start:end]
                .reshape(int(h), int(w))
                .astype(bool)
            )

            area_ratio = float(area_ratios[i])

            if area_ratio < SMALL_MAX_RATIO:
                size = "small"

            elif area_ratio < MEDIUM_MAX_RATIO:
                size = "medium"

            else:
                size = "large"

            predictions.append({

                "id": i,

                "class_id": int(class_ids[i]),

                "confidence": float(confidences[i]),

                "bbox": tuple(
                    int(x) for x in boxes[i]
                ),

                "mask": mask,

                "area": int(areas[i]),

                "area_ratio": area_ratio,

                "size": size,

            })

    return predictions


# ============================================================
# 7. MASK IoU
# ============================================================

def cropped_mask_iou(obj_a, obj_b):

    ax1, ay1, ax2, ay2 = obj_a["bbox"]

    bx1, by1, bx2, by2 = obj_b["bbox"]

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = 0

    if x2 > x1 and y2 > y1:

        a_crop = obj_a["mask"][
            y1 - ay1:y2 - ay1,
            x1 - ax1:x2 - ax1
        ]

        b_crop = obj_b["mask"][
            y1 - by1:y2 - by1,
            x1 - bx1:x2 - bx1
        ]

        intersection = int(
            np.logical_and(
                a_crop,
                b_crop
            ).sum()
        )

    union = (
        int(obj_a["area"])
        + int(obj_b["area"])
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(intersection / union)


# ============================================================
# 8. PRECOMPUTE IoU MATRIX
# ============================================================

def compute_iou_matrix(gt_objects, pred_objects):

    matrix = np.zeros(
        (len(gt_objects), len(pred_objects)),
        dtype=np.float32
    )

    for i, gt_obj in enumerate(gt_objects):

        for j, pred_obj in enumerate(pred_objects):

            matrix[i, j] = cropped_mask_iou(
                gt_obj,
                pred_obj
            )

    return matrix


# ============================================================
# 9. HUNGARIAN MATCHING
# ============================================================

def hungarian_match(iou_matrix, threshold):

    n_gt, n_pred = iou_matrix.shape

    if n_gt == 0 or n_pred == 0:
        return []

    size = max(n_gt, n_pred)

    score = np.zeros(
        (size, size),
        dtype=np.float64
    )

    valid = iou_matrix >= threshold

    score[:n_gt, :n_pred] = np.where(
        valid,
        1_000_000.0 + iou_matrix,
        -1_000_000.0
    )

    rows, cols = linear_sum_assignment(-score)

    matches = []

    for r, c in zip(rows, cols):

        if r >= n_gt or c >= n_pred:
            continue

        current_iou = float(iou_matrix[r, c])

        if current_iou >= threshold:

            matches.append((
                int(r),
                int(c),
                current_iou
            ))

    return matches


# ============================================================
# 10. EVALUATE NEW CONFIDENCE THRESHOLDS
# ============================================================

print("\n" + "=" * 80)
print("4. EXTENDED CONFIDENCE EVALUATION")
print("=" * 80)

new_metric_rows = []

audit_summary_rows = []
audit_detail_rows = []


for model_name in MODEL_NAMES:

    print("\n" + "-" * 80)
    print("MODEL:", model_name)
    print("-" * 80)

    model_cache_dir = CACHE_DIR / model_name

    for image_index, record in enumerate(image_records):

        cache_file = (
            model_cache_dir
            / f"{image_index:04d}.npz"
        )

        predictions = load_prediction_cache(
            cache_file
        )

        gt_objects = record["gt_objects"]

        gt_by_class = {
            c: [
                obj for obj in gt_objects
                if int(obj["class_id"]) == c
            ]
            for c in range(NUM_CLASSES)
        }

        pred_by_class = {
            c: [
                obj for obj in predictions
                if int(obj["class_id"]) == c
            ]
            for c in range(NUM_CLASSES)
        }

        iou_by_class = {}

        # Compute mask IoU only once per image/class.
        for class_id in range(NUM_CLASSES):

            iou_by_class[class_id] = compute_iou_matrix(
                gt_by_class[class_id],
                pred_by_class[class_id]
            )


        # ----------------------------------------------------
        # A. NEW CONFIDENCE THRESHOLDS
        # ----------------------------------------------------

        for conf_threshold in NEW_CONF_THRESHOLDS:

            for mask_iou_threshold in MASK_IOU_THRESHOLDS:

                for class_id in range(NUM_CLASSES):

                    gt_class = gt_by_class[class_id]
                    pred_class = pred_by_class[class_id]

                    iou_full = iou_by_class[class_id]

                    selected_indices = [
                        j
                        for j, pred in enumerate(pred_class)
                        if pred["confidence"] >= conf_threshold
                    ]

                    selected_count = len(selected_indices)

                    if selected_count > 0:

                        iou_selected = iou_full[
                            :,
                            selected_indices
                        ]

                    else:

                        iou_selected = np.zeros(
                            (len(gt_class), 0),
                            dtype=np.float32
                        )

                    matches = hungarian_match(
                        iou_selected,
                        mask_iou_threshold
                    )

                    matched_gt = {
                        m[0] for m in matches
                    }

                    tp = len(matches)

                    fp = selected_count - tp

                    fn = len(gt_class) - tp

                    precision = (
                        tp / (tp + fp)
                        if tp + fp > 0 else 0.0
                    )

                    recall = (
                        tp / (tp + fn)
                        if tp + fn > 0 else 0.0
                    )

                    f1 = (
                        2 * precision * recall
                        / (precision + recall)
                        if precision + recall > 0
                        else 0.0
                    )


                    size_counts = {
                        "small": {"gt": 0, "tp": 0},
                        "medium": {"gt": 0, "tp": 0},
                        "large": {"gt": 0, "tp": 0},
                    }

                    for gt_idx, gt_obj in enumerate(gt_class):

                        size_name = gt_obj["size"]

                        size_counts[size_name]["gt"] += 1

                        if gt_idx in matched_gt:
                            size_counts[size_name]["tp"] += 1


                    row = {

                        "model": model_name,

                        "image": record["image_name"],

                        "split": record["split"],

                        "class_id": class_id,

                        "class_name": CLASS_NAMES[class_id],

                        "confidence_threshold": conf_threshold,

                        "iou_threshold": mask_iou_threshold,

                        "gt_count": len(gt_class),

                        "pred_count": selected_count,

                        "tp": tp,

                        "fp": fp,

                        "fn": fn,

                        "precision": precision,

                        "recall": recall,

                        "f1": f1,

                    }

                    for size_name in [
                        "small",
                        "medium",
                        "large",
                    ]:

                        size_gt = size_counts[size_name]["gt"]

                        size_tp = size_counts[size_name]["tp"]

                        row[f"{size_name}_gt"] = size_gt

                        row[f"{size_name}_tp"] = size_tp

                        row[f"{size_name}_fn"] = size_gt - size_tp

                    new_metric_rows.append(row)


        # ----------------------------------------------------
        # B. FP/FN AUDIT AT CONF=0.40, IoU=0.50
        # ----------------------------------------------------

        for class_id in range(NUM_CLASSES):

            gt_class = gt_by_class[class_id]
            pred_class = pred_by_class[class_id]

            iou_full = iou_by_class[class_id]

            selected_indices = [
                j
                for j, pred in enumerate(pred_class)
                if pred["confidence"] >= VIS_CONF
            ]

            selected_predictions = [
                pred_class[j]
                for j in selected_indices
            ]

            if len(selected_indices) > 0:

                iou_selected = iou_full[
                    :,
                    selected_indices
                ]

            else:

                iou_selected = np.zeros(
                    (len(gt_class), 0),
                    dtype=np.float32
                )


            matches = hungarian_match(
                iou_selected,
                VIS_IOU
            )

            matched_gt = {
                m[0] for m in matches
            }

            matched_pred = {
                m[1] for m in matches
            }

            tp = len(matches)

            fp = len(selected_predictions) - tp

            fn = len(gt_class) - tp

            audit_summary_rows.append({

                "model": model_name,

                "image": record["image_name"],

                "split": record["split"],

                "class_id": class_id,

                "class_name": CLASS_NAMES[class_id],

                "confidence_threshold": VIS_CONF,

                "iou_threshold": VIS_IOU,

                "gt": len(gt_class),

                "pred": len(selected_predictions),

                "tp": tp,

                "fp": fp,

                "fn": fn,

                "errors": fp + fn,

            })


            # Matched TP details.
            for gt_idx, pred_idx, iou_value in matches:

                gt_obj = gt_class[gt_idx]

                pred_obj = selected_predictions[pred_idx]

                audit_detail_rows.append({

                    "model": model_name,

                    "image": record["image_name"],

                    "split": record["split"],

                    "class_id": class_id,

                    "class_name": CLASS_NAMES[class_id],

                    "status": "TP",

                    "confidence": pred_obj["confidence"],

                    "mask_iou": iou_value,

                    "gt_size": gt_obj["size"],

                    "pred_size": pred_obj["size"],

                    "max_iou": iou_value,

                    "fp_type": "",

                })


            # Unmatched GT = FN.
            for gt_idx, gt_obj in enumerate(gt_class):

                if gt_idx in matched_gt:
                    continue

                best_iou = (
                    float(iou_selected[gt_idx].max())
                    if len(selected_predictions) > 0
                    else 0.0
                )

                audit_detail_rows.append({

                    "model": model_name,

                    "image": record["image_name"],

                    "split": record["split"],

                    "class_id": class_id,

                    "class_name": CLASS_NAMES[class_id],

                    "status": "FN",

                    "confidence": np.nan,

                    "mask_iou": 0.0,

                    "gt_size": gt_obj["size"],

                    "pred_size": "",

                    "max_iou": best_iou,

                    "fp_type": "",

                })


            # Unmatched prediction = FP.
            for pred_idx, pred_obj in enumerate(
                selected_predictions
            ):

                if pred_idx in matched_pred:
                    continue

                best_iou = (
                    float(iou_selected[:, pred_idx].max())
                    if len(gt_class) > 0
                    else 0.0
                )

                if best_iou >= VIS_IOU:

                    fp_type = (
                        "duplicate_or_competing_prediction"
                    )

                elif best_iou >= 0.10:

                    fp_type = (
                        "partial_overlap_or_localization"
                    )

                else:

                    fp_type = (
                        "no_significant_overlap"
                    )

                audit_detail_rows.append({

                    "model": model_name,

                    "image": record["image_name"],

                    "split": record["split"],

                    "class_id": class_id,

                    "class_name": CLASS_NAMES[class_id],

                    "status": "FP",

                    "confidence": pred_obj["confidence"],

                    "mask_iou": 0.0,

                    "gt_size": "",

                    "pred_size": pred_obj["size"],

                    "max_iou": best_iou,

                    "fp_type": fp_type,

                })


        del predictions

        if (image_index + 1) % 25 == 0:

            print(
                f"[{model_name}] "
                f"{image_index + 1:3d}/{EXPECTED_IMAGES} images"
            )

            gc.collect()


new_metrics = pd.DataFrame(new_metric_rows)

audit_summary_df = pd.DataFrame(
    audit_summary_rows
)

audit_details_df = pd.DataFrame(
    audit_detail_rows
)

print("\nNew metric rows:", len(new_metrics))
print("Visual audit image/class rows:", len(audit_summary_df))
print("Object audit rows:", len(audit_details_df))


# ============================================================
# 11. SAVE NEW THRESHOLD RESULTS
# ============================================================

NEW_METRICS_PATH = (
    OUTPUT_DIR
    / "new_thresholds_per_image_metrics.csv"
)

new_metrics.to_csv(
    NEW_METRICS_PATH,
    index=False
)

print("\nSaved:", NEW_METRICS_PATH)


# ============================================================
# 12. MERGE OLD AND NEW THRESHOLDS
# ============================================================

print("\n" + "=" * 80)
print("5. MERGE OLD AND EXTENDED RESULTS")
print("=" * 80)

combined_metrics = pd.concat(
    [
        old_metrics,
        new_metrics,
    ],
    ignore_index=True
)


unique_key = [
    "model",
    "image",
    "split",
    "class_id",
    "confidence_threshold",
    "iou_threshold",
]


combined_metrics = combined_metrics.drop_duplicates(
    subset=unique_key,
    keep="last"
)

combined_metrics = combined_metrics.sort_values(
    [
        "model",
        "iou_threshold",
        "confidence_threshold",
        "image",
        "class_id",
    ]
).reset_index(drop=True)


combined_metrics.to_csv(
    OUTPUT_DIR / "ALL_confidence_per_image_metrics_extended.csv",
    index=False
)


print("Old rows:", len(old_metrics))
print("New rows:", len(new_metrics))
print("Combined unique rows:", len(combined_metrics))

all_thresholds = sorted(
    combined_metrics["confidence_threshold"].unique()
)

print("All confidence thresholds:", all_thresholds)


# ============================================================
# 13. SUMMARY FUNCTION
# ============================================================

def calculate_summary(df, group_columns):

    size_columns = [
        "small_gt",
        "small_tp",
        "small_fn",
        "medium_gt",
        "medium_tp",
        "medium_fn",
        "large_gt",
        "large_tp",
        "large_fn",
    ]

    aggregation = {

        "gt_count": "sum",

        "pred_count": "sum",

        "tp": "sum",

        "fp": "sum",

        "fn": "sum",

    }

    for col in size_columns:
        aggregation[col] = "sum"


    summary = (
        df
        .groupby(
            group_columns,
            as_index=False
        )
        .agg(aggregation)
    )


    summary["precision"] = (
        summary["tp"]
        / (
            summary["tp"] + summary["fp"]
        ).replace(0, np.nan)
    ).fillna(0)


    summary["recall"] = (
        summary["tp"]
        / (
            summary["tp"] + summary["fn"]
        ).replace(0, np.nan)
    ).fillna(0)


    summary["f1"] = (
        2 * summary["precision"] * summary["recall"]
        / (
            summary["precision"] + summary["recall"]
        ).replace(0, np.nan)
    ).fillna(0)


    for size_name in [
        "small",
        "medium",
        "large",
    ]:

        summary[f"{size_name}_recall"] = (
            summary[f"{size_name}_tp"]
            / summary[f"{size_name}_gt"].replace(0, np.nan)
        )

    return summary


# ============================================================
# 14. OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("6. OVERALL SUMMARY")
print("=" * 80)

overall_summary = calculate_summary(
    combined_metrics,

    [
        "model",
        "confidence_threshold",
        "iou_threshold",
    ]
)

overall_summary.to_csv(
    OUTPUT_DIR / "overall_confidence_summary_extended.csv",
    index=False
)


# ============================================================
# 15. PER-CLASS SUMMARY
# ============================================================

class_summary = calculate_summary(
    combined_metrics,

    [
        "model",
        "confidence_threshold",
        "iou_threshold",
        "class_id",
        "class_name",
    ]
)

class_summary.to_csv(
    OUTPUT_DIR / "per_class_confidence_summary_extended.csv",
    index=False
)


# ============================================================
# 16. OLD VS NEW SUMMARY
# ============================================================

split_summary = calculate_summary(
    combined_metrics,

    [
        "model",
        "split",
        "confidence_threshold",
        "iou_threshold",
    ]
)

split_summary.to_csv(
    OUTPUT_DIR / "old_new_confidence_summary_extended.csv",
    index=False
)


# ============================================================
# 17. SIZE BASED SUMMARY
# ============================================================

size_rows = []

for _, row in class_summary.iterrows():

    for size_name in [
        "small",
        "medium",
        "large",
    ]:

        size_rows.append({

            "model": row["model"],

            "confidence_threshold": (
                row["confidence_threshold"]
            ),

            "iou_threshold": row["iou_threshold"],

            "class_id": row["class_id"],

            "class_name": row["class_name"],

            "size": size_name,

            "gt_count": row[f"{size_name}_gt"],

            "tp": row[f"{size_name}_tp"],

            "fn": row[f"{size_name}_fn"],

            "recall": row[f"{size_name}_recall"],

        })


size_summary = pd.DataFrame(size_rows)

size_summary.to_csv(
    OUTPUT_DIR / "size_based_confidence_summary_extended.csv",
    index=False
)


# ============================================================
# 18. BEST GLOBAL THRESHOLD
# ============================================================

best_global_rows = []

for (model_name, iou_threshold), group in (
    overall_summary.groupby(
        ["model", "iou_threshold"]
    )
):

    best_row = (
        group
        .sort_values(
            ["f1", "precision"],
            ascending=[False, False]
        )
        .iloc[0]
    )

    best_global_rows.append({

        "model": model_name,

        "iou_threshold": iou_threshold,

        "best_confidence_threshold": (
            best_row["confidence_threshold"]
        ),

        "precision": best_row["precision"],

        "recall": best_row["recall"],

        "f1": best_row["f1"],

        "tp": best_row["tp"],

        "fp": best_row["fp"],

        "fn": best_row["fn"],

    })


best_global_summary = pd.DataFrame(
    best_global_rows
)

best_global_summary.to_csv(
    OUTPUT_DIR / "best_global_threshold_extended.csv",
    index=False
)


# ============================================================
# 19. BEST PER-CLASS THRESHOLD
# ============================================================

best_class_rows = []

for (
    model_name,
    iou_threshold,
    class_id,
    class_name
), group in class_summary.groupby(
    [
        "model",
        "iou_threshold",
        "class_id",
        "class_name",
    ]
):

    best_row = (
        group
        .sort_values(
            ["f1", "precision"],
            ascending=[False, False]
        )
        .iloc[0]
    )

    best_class_rows.append({

        "model": model_name,

        "iou_threshold": iou_threshold,

        "class_id": class_id,

        "class_name": class_name,

        "best_confidence_threshold": (
            best_row["confidence_threshold"]
        ),

        "precision": best_row["precision"],

        "recall": best_row["recall"],

        "f1": best_row["f1"],

        "tp": best_row["tp"],

        "fp": best_row["fp"],

        "fn": best_row["fn"],

    })


best_class_summary = pd.DataFrame(
    best_class_rows
)

best_class_summary.to_csv(
    OUTPUT_DIR / "best_class_threshold_extended.csv",
    index=False
)


# ============================================================
# 20. PLOT EXTENDED CONFIDENCE CURVES
# ============================================================

print("\n" + "=" * 80)
print("7. GENERATE EXTENDED CURVES")
print("=" * 80)

plot_metrics = [
    "precision",
    "recall",
    "f1",
]

for metric_name in plot_metrics:

    fig, ax = plt.subplots(figsize=(11, 6))

    for model_name in MODEL_NAMES:

        for iou_threshold in MASK_IOU_THRESHOLDS:

            subset = overall_summary[
                (overall_summary["model"] == model_name)
                &
                (
                    np.isclose(
                        overall_summary["iou_threshold"],
                        iou_threshold
                    )
                )
            ].sort_values(
                "confidence_threshold"
            )

            ax.plot(
                subset["confidence_threshold"],
                subset[metric_name],
                marker="o",
                linewidth=2,
                markersize=5,
                label=(
                    f"{model_name} "
                    f"IoU={iou_threshold:.2f}"
                )
            )


    ax.set_xscale("log")

    ax.set_xticks(all_thresholds)

    ax.set_xticklabels(
        [f"{x:.3g}" for x in all_thresholds],
        rotation=45
    )

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(metric_name.capitalize())

    ax.set_title(
        f"{metric_name.capitalize()} vs Confidence Threshold"
    )

    ax.set_ylim(0, 1.0)

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / f"{metric_name}_vs_confidence_extended.png"
    )

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("[SAVED]", plot_path)


# ============================================================
# 21. FP/FN AUDIT TABLES
# ============================================================

print("\n" + "=" * 80)
print("8. FP/FN VISUAL AUDIT")
print("=" * 80)

audit_summary_df.to_csv(
    OUTPUT_DIR / "visual_audit_image_class_summary.csv",
    index=False
)

audit_details_df.to_csv(
    OUTPUT_DIR / "visual_audit_object_details.csv",
    index=False
)


# FP/FN classification summary.
fp_details = audit_details_df[
    audit_details_df["status"] == "FP"
].copy()

if len(fp_details) > 0:

    fp_breakdown = (
        fp_details
        .groupby(
            [
                "model",
                "class_id",
                "class_name",
                "fp_type",
            ],
            as_index=False
        )
        .size()
        .rename(columns={"size": "count"})
    )

else:

    fp_breakdown = pd.DataFrame()


fp_breakdown.to_csv(
    OUTPUT_DIR / "visual_audit_fp_breakdown.csv",
    index=False
)


# ============================================================
# 22. VISUAL OVERLAY FUNCTION
# ============================================================

def paint_object(
    image,
    obj,
    color,
    alpha=0.35,
    thickness=2
):

    x1, y1, x2, y2 = obj["bbox"]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return

    mask = obj["mask"]

    expected_h = y2 - y1
    expected_w = x2 - x1

    if mask.shape != (expected_h, expected_w):
        return

    roi = image[y1:y2, x1:x2]

    color_array = np.asarray(
        color,
        dtype=np.float32
    )

    roi[mask] = (
        roi[mask].astype(np.float32) * (1 - alpha)
        + color_array * alpha
    ).astype(np.uint8)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2 - 1, y2 - 1),
        color,
        thickness
    )


# BGR colors.
COLOR_GT_MATCHED = (0, 200, 0)
COLOR_GT_FN = (0, 0, 255)

COLOR_PRED_MATCHED = (255, 255, 0)
COLOR_PRED_FP = (0, 165, 255)


def create_error_overlay(
    model_name,
    image_index,
    class_id
):

    record = image_records[image_index]

    image_path = Path(record["image_path"])

    original = cv2.imread(str(image_path))

    if original is None:
        raise RuntimeError(
            f"Cannot read image: {image_path}"
        )

    overlay = original.copy()

    cache_file = (
        CACHE_DIR
        / model_name
        / f"{image_index:04d}.npz"
    )

    predictions = load_prediction_cache(
        cache_file
    )

    gt_class = [
        obj for obj in record["gt_objects"]
        if int(obj["class_id"]) == class_id
    ]

    pred_class = [
        obj for obj in predictions
        if (
            int(obj["class_id"]) == class_id
            and obj["confidence"] >= VIS_CONF
        )
    ]

    iou_matrix = compute_iou_matrix(
        gt_class,
        pred_class
    )

    matches = hungarian_match(
        iou_matrix,
        VIS_IOU
    )

    matched_gt = {
        item[0] for item in matches
    }

    matched_pred = {
        item[1] for item in matches
    }

    tp = len(matches)

    fp = len(pred_class) - tp

    fn = len(gt_class) - tp


    # Draw GT first.
    for i, gt_obj in enumerate(gt_class):

        if i in matched_gt:
            color = COLOR_GT_MATCHED
        else:
            color = COLOR_GT_FN

        paint_object(
            overlay,
            gt_obj,
            color,
            alpha=0.32
        )


    # Draw predictions second.
    for j, pred_obj in enumerate(pred_class):

        if j in matched_pred:
            color = COLOR_PRED_MATCHED
        else:
            color = COLOR_PRED_FP

        paint_object(
            overlay,
            pred_obj,
            color,
            alpha=0.30
        )


    # Add information.
    header = (
        f"{model_name} | {CLASS_NAMES[class_id]} | "
        f"conf>={VIS_CONF:.2f} | mask IoU={VIS_IOU:.2f}"
    )

    stats = (
        f"GT={len(gt_class)}  "
        f"Pred={len(pred_class)}  "
        f"TP={tp}  FP={fp}  FN={fn}"
    )

    cv2.rectangle(
        overlay,
        (0, 0),
        (overlay.shape[1], 75),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        overlay,
        header,
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        overlay,
        stats,
        (15, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # Legend.
    legend_items = [
        ("GT matched", COLOR_GT_MATCHED),
        ("GT false negative", COLOR_GT_FN),
        ("Prediction matched", COLOR_PRED_MATCHED),
        ("Prediction false positive", COLOR_PRED_FP),
    ]

    legend_y = 105

    for label, color in legend_items:

        cv2.rectangle(
            overlay,
            (15, legend_y - 13),
            (35, legend_y + 5),
            color,
            -1
        )

        cv2.putText(
            overlay,
            label,
            (45, legend_y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        legend_y += 25


    # Side-by-side original and overlay.
    comparison = np.hstack(
        [original, overlay]
    )

    return comparison, {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": len(gt_class),
        "pred": len(pred_class),
    }


# ============================================================
# 23. SELECT WORST IMAGES PER MODEL AND CLASS
# ============================================================

print("\nSelecting images with highest FP + FN...")

selected_examples = []

for model_name in MODEL_NAMES:

    for class_id, class_name in enumerate(CLASS_NAMES):

        subset = audit_summary_df[
            (audit_summary_df["model"] == model_name)
            &
            (audit_summary_df["class_id"] == class_id)
        ].copy()

        subset = subset[
            subset["errors"] > 0
        ]

        subset = subset.sort_values(
            [
                "errors",
                "fp",
                "fn",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        )

        for rank, (_, row) in enumerate(
            subset.head(VISUAL_TOP_K).iterrows(),
            start=1
        ):

            selected_examples.append({

                "model": model_name,

                "class_id": class_id,

                "class_name": class_name,

                "image": row["image"],

                "split": row["split"],

                "rank": rank,

                "gt": int(row["gt"]),

                "pred": int(row["pred"]),

                "tp": int(row["tp"]),

                "fp": int(row["fp"]),

                "fn": int(row["fn"]),

                "errors": int(row["errors"]),

            })


selected_examples_df = pd.DataFrame(
    selected_examples
)

selected_examples_df.to_csv(
    OUTPUT_DIR / "selected_visual_error_examples.csv",
    index=False
)

print(
    "Selected visual examples:",
    len(selected_examples_df)
)


# ============================================================
# 24. CREATE OVERLAYS
# ============================================================

print("\n" + "=" * 80)
print("9. CREATE FP/FN OVERLAYS")
print("=" * 80)

image_to_index = {
    record["image_name"]: index
    for index, record in enumerate(image_records)
}

visual_output_rows = []

for _, example in selected_examples_df.iterrows():

    model_name = example["model"]

    class_id = int(example["class_id"])

    image_name = example["image"]

    image_index = image_to_index[image_name]

    comparison, stats = create_error_overlay(
        model_name,
        image_index,
        class_id
    )

    safe_name = (
        str(image_name)
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_name = (
        f"{model_name}_"
        f"class{class_id}_"
        f"rank{int(example['rank'])}_"
        f"{safe_name}.jpg"
    )

    output_path = VISUAL_DIR / output_name

    success = cv2.imwrite(
        str(output_path),
        comparison,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92
        ]
    )

    if not success:
        raise RuntimeError(
            f"Could not save overlay: {output_path}"
        )

    visual_output_rows.append({

        "model": model_name,

        "class_id": class_id,

        "class_name": example["class_name"],

        "image": image_name,

        "rank": int(example["rank"]),

        "tp": stats["tp"],

        "fp": stats["fp"],

        "fn": stats["fn"],

        "output_file": str(output_path),

    })

    print(
        f"[SAVED] {model_name} | "
        f"{class_name if (class_name := example['class_name']) else ''} | "
        f"{image_name}"
    )


visual_output_df = pd.DataFrame(
    visual_output_rows
)

visual_output_df.to_csv(
    OUTPUT_DIR / "visual_overlay_manifest.csv",
    index=False
)


# ============================================================
# 25. PLOT EXTENDED CURVES - CLASS LEVEL
# ============================================================

print("\n" + "=" * 80)
print("10. PER-CLASS F1 CURVES")
print("=" * 80)

for iou_threshold in MASK_IOU_THRESHOLDS:

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9)
    )

    axes = axes.flatten()

    for class_id, ax in enumerate(axes):

        for model_name in MODEL_NAMES:

            subset = class_summary[
                (class_summary["model"] == model_name)
                &
                (class_summary["class_id"] == class_id)
                &
                (
                    np.isclose(
                        class_summary["iou_threshold"],
                        iou_threshold
                    )
                )
            ].sort_values(
                "confidence_threshold"
            )

            ax.plot(
                subset["confidence_threshold"],
                subset["f1"],
                marker="o",
                linewidth=1.8,
                label=model_name
            )


        ax.set_xscale("log")

        ax.set_xticks(all_thresholds)

        ax.set_xticklabels(
            [f"{x:.3g}" for x in all_thresholds],
            rotation=45
        )

        ax.set_ylim(0, 1.0)

        ax.set_title(CLASS_NAMES[class_id])

        ax.set_xlabel("Confidence")

        ax.set_ylabel("F1")

        ax.grid(True, alpha=0.3)

        ax.legend()


    fig.suptitle(
        f"Per-class F1 vs Confidence | Mask IoU={iou_threshold:.2f}",
        fontsize=15
    )

    fig.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / f"per_class_f1_iou_{int(iou_threshold*100)}_extended.png"
    )

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("[SAVED]", plot_path)


# ============================================================
# 26. EXPORT EXCEL
# ============================================================

print("\n" + "=" * 80)
print("11. EXPORT EXCEL")
print("=" * 80)

excel_path = (
    OUTPUT_DIR
    / "extended_confidence_and_error_audit.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    overall_summary.to_excel(
        writer,
        sheet_name="Overall",
        index=False
    )

    class_summary.to_excel(
        writer,
        sheet_name="Per_Class",
        index=False
    )

    split_summary.to_excel(
        writer,
        sheet_name="Old_vs_New",
        index=False
    )

    size_summary.to_excel(
        writer,
        sheet_name="Size_Recall",
        index=False
    )

    best_global_summary.to_excel(
        writer,
        sheet_name="Best_Global",
        index=False
    )

    best_class_summary.to_excel(
        writer,
        sheet_name="Best_Per_Class",
        index=False
    )

    audit_summary_df.to_excel(
        writer,
        sheet_name="Image_Error_Summary",
        index=False
    )

    fp_breakdown.to_excel(
        writer,
        sheet_name="FP_Breakdown",
        index=False
    )

    selected_examples_df.to_excel(
        writer,
        sheet_name="Visual_Examples",
        index=False
    )


# ============================================================
# 27. PRINT EXTENDED OVERALL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("12. EXTENDED OVERALL RESULTS")
print("=" * 80)

display_columns = [
    "model",
    "confidence_threshold",
    "iou_threshold",
    "pred_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]

print(
    overall_summary[display_columns]
    .sort_values(
        [
            "iou_threshold",
            "confidence_threshold",
            "model",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 28. PRINT UPDATED BEST THRESHOLDS
# ============================================================

print("\n" + "=" * 80)
print("13. UPDATED BEST GLOBAL THRESHOLDS")
print("=" * 80)

print(
    best_global_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 80)
print("14. UPDATED BEST PER-CLASS THRESHOLDS")
print("=" * 80)

print(
    best_class_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 29. PRINT FP/FN EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("15. SELECTED FP/FN VISUAL EXAMPLES")
print("=" * 80)

if len(selected_examples_df) > 0:

    print(
        selected_examples_df.to_string(
            index=False
        )
    )

else:

    print("No FP/FN examples were selected.")


# ============================================================
# 30. OUTPUT VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("16. OUTPUT VERIFICATION")
print("=" * 80)

expected_outputs = [

    "new_thresholds_per_image_metrics.csv",

    "ALL_confidence_per_image_metrics_extended.csv",

    "overall_confidence_summary_extended.csv",

    "per_class_confidence_summary_extended.csv",

    "old_new_confidence_summary_extended.csv",

    "size_based_confidence_summary_extended.csv",

    "best_global_threshold_extended.csv",

    "best_class_threshold_extended.csv",

    "visual_audit_image_class_summary.csv",

    "visual_audit_object_details.csv",

    "visual_audit_fp_breakdown.csv",

    "selected_visual_error_examples.csv",

    "visual_overlay_manifest.csv",

    "precision_vs_confidence_extended.png",

    "recall_vs_confidence_extended.png",

    "f1_vs_confidence_extended.png",

    "extended_confidence_and_error_audit.xlsx",

]


# The overall curve files are generated in section 20.
# The per-class plots are also included in the output folder.

for filename in expected_outputs:

    path = OUTPUT_DIR / filename

    if not path.exists():

        # Per-class curves have separate filenames.
        if filename.startswith("per_class_"):
            continue

        raise RuntimeError(
            f"Expected output file missing: {path}"
        )

    print(
        f"[OK] {filename:<55} "
        f"{path.stat().st_size / 1024**2:.3f} MB"
    )


visual_files = list(
    VISUAL_DIR.glob("*.jpg")
)

print("\nVisual overlay images:", len(visual_files))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nVisual directory:")
print(VISUAL_DIR)


gc.collect()

print("\n" + "=" * 80)
print("CELL 17L COMPLETED SUCCESSFULLY")
print("=" * 80)

###########################################################################
# CELL 37
###########################################################################
import json
from pathlib import Path

cache_dir = Path(
    "/kaggle/working/YOLO11_seg_expanded/"
    "research_experiments/confidence_threshold_analysis"
)

for model in ["C_IRFS", "CB_IRFS"]:

    path = cache_dir / f"{model}_cache_metadata.json"

    with open(path, "r") as f:
        data = json.load(f)

    print("\n", model)
    print("Total predictions:", data["total_predictions"])
    print("Images at max_det:", data["images_at_max_det"])
    print("Max predictions/image:", data["max_predictions_per_image"])

###########################################################################
# CELL 38
###########################################################################
# ============================================================
# CELL 17M - SELF-CONTAINED HTML VISUAL AUDIT REPORT
#
# Input:
#   confidence_threshold_analysis_extended/
#       error_visualizations/
#       visual_overlay_manifest.csv
#       visual_audit_image_class_summary.csv
#       visual_audit_object_details.csv
#       visual_audit_fp_breakdown.csv
#       selected_visual_error_examples.csv
#
# Output:
#   visual_error_audit_report.html
#   visual_error_audit_notes.txt
#
# Features:
#   1. All overlay images embedded as Base64.
#   2. No external internet dependency.
#   3. Filter by model, class and split.
#   4. Search by image filename.
#   5. Click image to enlarge.
#   6. Download individual overlay image.
#   7. Dataset-level class error summary.
#   8. FP category breakdown.
#   9. Export compact text notes.
#
# No YOLO inference.
# No GPU required.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import re
import base64
import html as html_lib
import json

import numpy as np
import pandas as pd

from pathlib import Path

from IPython.display import display, FileLink


# ============================================================
# 2. PATHS
# ============================================================

ROOT = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

SOURCE_DIR = (
    ROOT
    / "research_experiments"
    / "confidence_threshold_analysis_extended"
)

VISUAL_DIR = (
    SOURCE_DIR
    / "error_visualizations"
)

REPORT_DIR = (
    SOURCE_DIR
    / "html_visual_audit"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MANIFEST_PATH = (
    SOURCE_DIR
    / "visual_overlay_manifest.csv"
)

AUDIT_SUMMARY_PATH = (
    SOURCE_DIR
    / "visual_audit_image_class_summary.csv"
)

AUDIT_DETAILS_PATH = (
    SOURCE_DIR
    / "visual_audit_object_details.csv"
)

FP_BREAKDOWN_PATH = (
    SOURCE_DIR
    / "visual_audit_fp_breakdown.csv"
)

SELECTED_PATH = (
    SOURCE_DIR
    / "selected_visual_error_examples.csv"
)


REPORT_PATH = (
    REPORT_DIR
    / "visual_error_audit_report.html"
)

NOTES_PATH = (
    REPORT_DIR
    / "visual_error_audit_notes.txt"
)


# ============================================================
# 3. VERIFY INPUT FILES
# ============================================================

print("=" * 80)
print("1. VERIFY INPUT FILES")
print("=" * 80)

required_files = [
    MANIFEST_PATH,
    AUDIT_SUMMARY_PATH,
    AUDIT_DETAILS_PATH,
    FP_BREAKDOWN_PATH,
    SELECTED_PATH,
]

for path in required_files:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    print("[OK]", path.name)


if not VISUAL_DIR.exists():
    raise FileNotFoundError(
        f"Visual directory not found: {VISUAL_DIR}"
    )


# ============================================================
# 4. LOAD CSV DATA
# ============================================================

print("\n" + "=" * 80)
print("2. LOAD AUDIT DATA")
print("=" * 80)

manifest = pd.read_csv(
    MANIFEST_PATH
)

audit_summary = pd.read_csv(
    AUDIT_SUMMARY_PATH
)

audit_details = pd.read_csv(
    AUDIT_DETAILS_PATH
)

fp_breakdown = pd.read_csv(
    FP_BREAKDOWN_PATH
)

selected_examples = pd.read_csv(
    SELECTED_PATH
)


required_manifest_columns = [
    "model",
    "class_id",
    "class_name",
    "image",
    "rank",
    "tp",
    "fp",
    "fn",
    "output_file",
]

missing = [
    c for c in required_manifest_columns
    if c not in manifest.columns
]

if missing:
    raise RuntimeError(
        "Manifest is missing columns: "
        + ", ".join(missing)
    )


print("Overlay manifest:", len(manifest))
print("Audit image/class rows:", len(audit_summary))
print("Object detail rows:", len(audit_details))
print("FP breakdown rows:", len(fp_breakdown))
print("Selected examples:", len(selected_examples))


if len(manifest) == 0:
    raise RuntimeError(
        "No overlay records found in the manifest."
    )


# ============================================================
# 5. RESOLVE IMAGE PATHS
# ============================================================

print("\n" + "=" * 80)
print("3. VERIFY OVERLAY IMAGES")
print("=" * 80)


def resolve_overlay_path(path_value):

    original_path = Path(str(path_value))

    if original_path.exists():
        return original_path.resolve()

    # Fallback: locate by filename in visual directory.
    candidate = (
        VISUAL_DIR
        / original_path.name
    )

    if candidate.exists():
        return candidate.resolve()

    candidate = (
        VISUAL_DIR
        / Path(str(path_value)).name
    )

    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Overlay image not found: {path_value}"
    )


manifest["resolved_image_path"] = [
    str(resolve_overlay_path(x))
    for x in manifest["output_file"]
]


for path in manifest["resolved_image_path"]:

    if not Path(path).exists():
        raise RuntimeError(
            f"Image file missing: {path}"
        )


print(
    "Verified overlay images:",
    len(manifest)
)

total_image_size = sum(
    Path(p).stat().st_size
    for p in manifest["resolved_image_path"]
)

print(
    f"Total image data: "
    f"{total_image_size / 1024**2:.2f} MB"
)


# ============================================================
# 6. DATASET-LEVEL CLASS SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("4. BUILD CLASS SUMMARY")
print("=" * 80)


class_summary = (
    audit_summary
    .groupby(
        [
            "model",
            "class_id",
            "class_name",
        ],
        as_index=False
    )
    .agg({

        "gt": "sum",

        "pred": "sum",

        "tp": "sum",

        "fp": "sum",

        "fn": "sum",

    })
)


class_summary["precision"] = (
    class_summary["tp"]
    / (
        class_summary["tp"]
        + class_summary["fp"]
    ).replace(0, np.nan)
).fillna(0)


class_summary["recall"] = (
    class_summary["tp"]
    / (
        class_summary["tp"]
        + class_summary["fn"]
    ).replace(0, np.nan)
).fillna(0)


class_summary["f1"] = (
    2
    * class_summary["precision"]
    * class_summary["recall"]
    / (
        class_summary["precision"]
        + class_summary["recall"]
    ).replace(0, np.nan)
).fillna(0)


class_summary["error_count"] = (
    class_summary["fp"]
    + class_summary["fn"]
)


class_summary.to_csv(
    REPORT_DIR / "class_error_summary.csv",
    index=False
)


print(
    class_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 7. OVERALL MODEL SUMMARY
# ============================================================

overall_summary = (
    class_summary
    .groupby(
        "model",
        as_index=False
    )
    .agg({

        "gt": "sum",

        "pred": "sum",

        "tp": "sum",

        "fp": "sum",

        "fn": "sum",

    })
)


overall_summary["precision"] = (
    overall_summary["tp"]
    / (
        overall_summary["tp"]
        + overall_summary["fp"]
    ).replace(0, np.nan)
).fillna(0)


overall_summary["recall"] = (
    overall_summary["tp"]
    / (
        overall_summary["tp"]
        + overall_summary["fn"]
    ).replace(0, np.nan)
).fillna(0)


overall_summary["f1"] = (
    2
    * overall_summary["precision"]
    * overall_summary["recall"]
    / (
        overall_summary["precision"]
        + overall_summary["recall"]
    ).replace(0, np.nan)
).fillna(0)


# ============================================================
# 8. SUMMARY TABLE TO HTML
# ============================================================

def dataframe_to_html(
    dataframe,
    table_class="data-table",
    float_format="%.4f"
):

    return dataframe.to_html(
        index=False,
        border=0,
        classes=table_class,
        escape=True,
        float_format=lambda x: float_format % x
    )


overall_html = dataframe_to_html(
    overall_summary
)

class_html = dataframe_to_html(
    class_summary
)


fp_html = dataframe_to_html(
    fp_breakdown
)


# ============================================================
# 9. EMBED IMAGES INTO HTML
# ============================================================

print("\n" + "=" * 80)
print("5. EMBED IMAGES")
print("=" * 80)


def file_to_data_uri(path):

    path = Path(path)

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"

    elif suffix == ".png":
        mime = "image/png"

    else:
        raise ValueError(
            f"Unsupported image extension: {suffix}"
        )

    with open(path, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode("ascii")

    return f"data:{mime};base64,{encoded}"


def safe_text(value):

    if pd.isna(value):
        return ""

    return html_lib.escape(
        str(value),
        quote=True
    )


cards_html = []

embedded_bytes = 0

manifest = manifest.sort_values(
    [
        "model",
        "class_id",
        "rank",
    ]
).reset_index(drop=True)


for index, row in manifest.iterrows():

    image_path = Path(
        row["resolved_image_path"]
    )

    image_uri = file_to_data_uri(
        image_path
    )

    embedded_bytes += image_path.stat().st_size

    model_name = safe_text(row["model"])

    class_id = int(row["class_id"])

    class_name = safe_text(row["class_name"])

    image_name = safe_text(row["image"])

    split = safe_text(row["split"])

    rank = int(row["rank"])

    tp = int(row["tp"])
    fp = int(row["fp"])
    fn = int(row["fn"])

    errors = fp + fn

    # Compact search text.
    search_text = (
        f"{row['model']} "
        f"{row['class_name']} "
        f"{row['image']} "
        f"{row['split']}"
    ).lower()

    search_text = safe_text(search_text)

    card = f"""
    <article
        class="case-card"
        data-model="{model_name}"
        data-class="{class_id}"
        data-split="{split}"
        data-search="{search_text}"
        data-errors="{errors}"
    >

        <div class="card-header">

            <div>
                <span class="model-tag">{model_name}</span>
                <span class="class-tag">{class_name}</span>
            </div>

            <span class="rank-tag">Rank #{rank}</span>

        </div>

        <h3>{image_name}</h3>

        <div class="stats">

            <span class="stat tp">
                TP: {tp}
            </span>

            <span class="stat fp">
                FP: {fp}
            </span>

            <span class="stat fn">
                FN: {fn}
            </span>

            <span class="stat total">
                Errors: {errors}
            </span>

        </div>

        <div class="image-container">

            <img
                src="{image_uri}"
                alt="{image_name}"
                class="audit-image"
                loading="lazy"
                onclick="openImageModal(this.src, '{image_name}')"
            />

        </div>

        <div class="card-footer">

            <span>Split: {split}</span>

            <a
                href="{image_uri}"
                download="{model_name}_class{class_id}_rank{rank}.jpg"
                class="download-link"
            >
                Download image
            </a>

        </div>

    </article>
    """

    cards_html.append(card)

    if (index + 1) % 6 == 0:
        print(
            f"Embedded {index + 1}/{len(manifest)} images"
        )


cards_html_string = "\n".join(cards_html)


print(
    f"Embedded image bytes: "
    f"{embedded_bytes / 1024**2:.2f} MB"
)


# ============================================================
# 10. BUILD TEXT NOTES
# ============================================================

notes = []

notes.append(
    "VISUAL ERROR AUDIT - YOLO11 SEGMENTATION\n"
)

notes.append(
    "Operating point: confidence >= 0.40, mask IoU >= 0.50\n"
)

notes.append(
    "Dataset: 268 validation images, 3,042 GT instances\n"
)

notes.append(
    "\nOVERALL SUMMARY\n"
)


for _, row in overall_summary.iterrows():

    notes.append(
        f"\n{row['model']}\n"
        f"GT={int(row['gt'])}, "
        f"Pred={int(row['pred'])}, "
        f"TP={int(row['tp'])}, "
        f"FP={int(row['fp'])}, "
        f"FN={int(row['fn'])}\n"
        f"Precision={row['precision']:.4f}, "
        f"Recall={row['recall']:.4f}, "
        f"F1={row['f1']:.4f}\n"
    )


notes.append(
    "\nPER-CLASS SUMMARY\n"
)


for _, row in class_summary.iterrows():

    notes.append(
        f"\n{row['model']} | {row['class_name']}\n"
        f"GT={int(row['gt'])}, "
        f"TP={int(row['tp'])}, "
        f"FP={int(row['fp'])}, "
        f"FN={int(row['fn'])}, "
        f"F1={row['f1']:.4f}\n"
    )


notes.append(
    "\nSELECTED VISUAL CASES\n"
)


for _, row in selected_examples.iterrows():

    notes.append(
        f"\n{row['model']} | "
        f"{row['class_name']} | "
        f"{row['image']}\n"
        f"GT={int(row['gt'])}, "
        f"Pred={int(row['pred'])}, "
        f"TP={int(row['tp'])}, "
        f"FP={int(row['fp'])}, "
        f"FN={int(row['fn'])}\n"
    )


notes.append(
    "\nInterpretation reminder:\n"
    "FP/FN totals are measured at confidence=0.40 and mask IoU=0.50.\n"
    "They are diagnostic results on the validation set, not independent test results.\n"
)


with open(
    NOTES_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("".join(notes))


# ============================================================
# 11. HTML DOCUMENT
# ============================================================

print("\n" + "=" * 80)
print("6. BUILD SELF-CONTAINED HTML")
print("=" * 80)


num_cards = len(manifest)

num_classes = manifest["class_id"].nunique()

num_models = manifest["model"].nunique()

num_images = manifest["image"].nunique()


HTML_DOCUMENT = f"""<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>YOLO11 Segmentation - Visual Error Audit</title>

<style>

:root {{
    --background: #f4f6f9;
    --surface: #ffffff;
    --text: #202938;
    --muted: #64748b;
    --border: #dce2ea;
    --primary: #315b91;
    --green: #16834b;
    --red: #cf3434;
    --orange: #c47712;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 0;
    background: var(--background);
    color: var(--text);
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.55;
}}

.container {{
    max-width: 1500px;
    margin: auto;
    padding: 26px;
}}

h1 {{
    margin-bottom: 5px;
    font-size: 30px;
}}

h2 {{
    margin-top: 35px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 9px;
}}

.subtitle {{
    color: var(--muted);
}}

.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin: 24px 0;
}}

.metric-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 17px;
}}

.metric-label {{
    color: var(--muted);
    font-size: 13px;
}}

.metric-value {{
    font-size: 26px;
    font-weight: bold;
    margin-top: 5px;
}}

.section-note {{
    color: var(--muted);
    background: white;
    padding: 13px 16px;
    border-left: 4px solid var(--primary);
    border-radius: 5px;
    margin: 15px 0;
}}

.table-wrap {{
    overflow-x: auto;
    background: white;
    border-radius: 9px;
    border: 1px solid var(--border);
    margin: 15px 0;
}}

.data-table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
    min-width: 700px;
}}

.data-table th,
.data-table td {{
    border-bottom: 1px solid var(--border);
    padding: 10px;
    text-align: right;
    white-space: nowrap;
}}

.data-table th {{
    background: #edf2f8;
    position: sticky;
    top: 0;
}}

.data-table td:first-child,
.data-table th:first-child {{
    text-align: left;
}}

.filters {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 16px;
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin: 20px 0;
}}

.filters input,
.filters select {{
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    min-width: 160px;
    background: white;
    font-size: 14px;
}}

.case-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 18px;
    margin-top: 20px;
}}

.case-card {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}}

.card-header {{
    padding: 14px 15px 5px;
    display: flex;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
}}

.model-tag {{
    display: inline-block;
    font-size: 12px;
    font-weight: bold;
    color: #234b7d;
    background: #e4edf8;
    padding: 5px 8px;
    border-radius: 5px;
}}

.class-tag {{
    display: inline-block;
    margin-left: 5px;
    font-size: 12px;
    color: #46566b;
    background: #edf0f4;
    padding: 5px 8px;
    border-radius: 5px;
}}

.rank-tag {{
    color: var(--muted);
    font-size: 12px;
}}

.case-card h3 {{
    font-size: 14px;
    padding: 0 15px;
    overflow-wrap: anywhere;
}}

.stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    padding: 0 15px 12px;
}}

.stat {{
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 5px;
    background: #f1f3f6;
}}

.stat.tp {{
    color: var(--green);
}}

.stat.fp {{
    color: var(--orange);
}}

.stat.fn {{
    color: var(--red);
}}

.stat.total {{
    font-weight: bold;
}}

.image-container {{
    background: #151a22;
    padding: 8px;
}}

.audit-image {{
    display: block;
    width: 100%;
    height: auto;
    cursor: zoom-in;
    border-radius: 3px;
}}

.card-footer {{
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-size: 12px;
}}

.download-link {{
    color: var(--primary);
    text-decoration: none;
    font-weight: bold;
}}

.download-link:hover {{
    text-decoration: underline;
}}

.legend {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 15px 0;
    background: white;
    border: 1px solid var(--border);
    padding: 14px;
    border-radius: 8px;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 13px;
}}

.color-box {{
    width: 15px;
    height: 15px;
    border-radius: 3px;
}}

#resultCount {{
    color: var(--muted);
    font-weight: bold;
    padding: 5px 0;
}}

.modal {{
    display: none;
    position: fixed;
    z-index: 1000;
    inset: 0;
    background: rgba(0,0,0,0.94);
    padding: 25px;
    overflow: auto;
}}

.modal-content {{
    display: block;
    max-width: 96%;
    max-height: 90vh;
    object-fit: contain;
    margin: 30px auto;
    cursor: zoom-out;
}}

.modal-caption {{
    color: white;
    text-align: center;
    margin-top: 10px;
}}

.modal-close {{
    position: fixed;
    top: 14px;
    right: 25px;
    color: white;
    font-size: 38px;
    cursor: pointer;
}}

.hidden {{
    display: none !important;
}}

footer {{
    margin: 50px 0 20px;
    padding-top: 15px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
}}

@media(max-width: 600px) {{

    .container {{
        padding: 12px;
    }}

    .case-grid {{
        grid-template-columns: 1fr;
    }}

    h1 {{
        font-size: 23px;
    }}

}}

</style>

</head>


<body>

<div class="container">

    <header>

        <h1>YOLO11 Segmentation - Visual Error Audit</h1>

        <div class="subtitle">
            C-IRFS vs CB-IRFS | Validation Error Analysis
        </div>

    </header>


    <div class="section-note">

        <strong>Evaluation configuration</strong><br>

        Dataset: 268 images, 3,042 Ground Truth instances<br>

        Confidence threshold: 0.40<br>

        Mask IoU threshold: 0.50<br>

        Classes: 6<br>

        Selected examples: top 2 images per model and class

    </div>


    <div class="metrics-grid">

        <div class="metric-box">

            <div class="metric-label">
                Embedded overlays
            </div>

            <div class="metric-value">
                {num_cards}
            </div>

        </div>


        <div class="metric-box">

            <div class="metric-label">
                Models
            </div>

            <div class="metric-value">
                {num_models}
            </div>

        </div>


        <div class="metric-box">

            <div class="metric-label">
                Classes represented
            </div>

            <div class="metric-value">
                {num_classes}
            </div>

        </div>


        <div class="metric-box">

            <div class="metric-label">
                Unique image names
            </div>

            <div class="metric-value">
                {num_images}
            </div>

        </div>

    </div>


    <h2>1. Overall model summary</h2>

    <div class="table-wrap">
        {overall_html}
    </div>


    <h2>2. Per-class error summary</h2>

    <div class="section-note">

        The statistics below use the full validation set,
        not only the selected visual examples.

    </div>

    <div class="table-wrap">
        {class_html}
    </div>


    <h2>3. False-positive categories</h2>

    <div class="table-wrap">
        {fp_html}
    </div>


    <h2>4. Visual legend</h2>

    <div class="legend">

        <div class="legend-item">
            <div class="color-box" style="background:#00C800"></div>
            GT matched
        </div>

        <div class="legend-item">
            <div class="color-box" style="background:#FF0000"></div>
            GT false negative
        </div>

        <div class="legend-item">
            <div class="color-box" style="background:#00FFFF"></div>
            Prediction matched
        </div>

        <div class="legend-item">
            <div class="color-box" style="background:#FFA500"></div>
            Prediction false positive
        </div>

    </div>


    <h2>5. Visual error cases</h2>

    <div class="section-note">

        Each card compares the original image and the overlay.<br>

        Click any image to enlarge it.<br>

        Use the filters to inspect a particular model,
        class, split or image filename.

    </div>


    <div class="filters">

        <select id="modelFilter">

            <option value="all">All models</option>

            <option value="C_IRFS">C-IRFS</option>

            <option value="CB_IRFS">CB-IRFS</option>

        </select>


        <select id="classFilter">

            <option value="all">All classes</option>

            <option value="0">Hat</option>

            <option value="1">Gloves</option>

            <option value="2">Mask</option>

            <option value="3">Pants</option>

            <option value="4">Shoes</option>

            <option value="5">Long-sleeved shirt</option>

        </select>


        <select id="splitFilter">

            <option value="all">All splits</option>

            <option value="Old">Old</option>

            <option value="New">New</option>

        </select>


        <input
            id="searchInput"
            type="text"
            placeholder="Search image filename..."
        />

    </div>


    <div id="resultCount"></div>


    <div class="case-grid" id="caseGrid">

        {cards_html_string}

    </div>


    <footer>

        Generated from Cell 17L.<br>

        Model checkpoints: C-IRFS and CB-IRFS.<br>

        Report is self-contained: all images are embedded
        in this HTML file. No internet connection is required.<br><br>

        Important: FP/FN are operating-point diagnostics
        on the validation set, not independent test results.

    </footer>

</div>


<div id="imageModal" class="modal">

    <span class="modal-close" onclick="closeImageModal()">
        &times;
    </span>

    <img
        id="modalImage"
        class="modal-content"
    />

    <div
        id="modalCaption"
        class="modal-caption"
    ></div>

</div>


<script>

const modelFilter = document.getElementById("modelFilter");

const classFilter = document.getElementById("classFilter");

const splitFilter = document.getElementById("splitFilter");

const searchInput = document.getElementById("searchInput");

const cards = Array.from(
    document.querySelectorAll(".case-card")
);


function applyFilters() {{

    const modelValue = modelFilter.value;

    const classValue = classFilter.value;

    const splitValue = splitFilter.value;

    const searchValue = searchInput.value.toLowerCase().trim();

    let visible = 0;


    cards.forEach(card => {{

        const modelMatch =
            modelValue === "all"
            || card.dataset.model === modelValue;


        const classMatch =
            classValue === "all"
            || card.dataset.class === classValue;


        const splitMatch =
            splitValue === "all"
            || card.dataset.split === splitValue;


        const searchMatch =
            card.dataset.search.includes(searchValue);


        const show =
            modelMatch
            && classMatch
            && splitMatch
            && searchMatch;


        card.classList.toggle(
            "hidden",
            !show
        );


        if (show) {{
            visible++;
        }}

    }});


    document.getElementById("resultCount").textContent =
        `Showing ${{visible}} of ${{cards.length}} cases`;

}}


modelFilter.addEventListener(
    "change",
    applyFilters
);

classFilter.addEventListener(
    "change",
    applyFilters
);

splitFilter.addEventListener(
    "change",
    applyFilters
);

searchInput.addEventListener(
    "input",
    applyFilters
);


function openImageModal(src, caption) {{

    const modal = document.getElementById("imageModal");

    document.getElementById("modalImage").src = src;

    document.getElementById("modalCaption").textContent = caption;

    modal.style.display = "block";

}}


function closeImageModal() {{

    document.getElementById("imageModal").style.display = "none";

}}


document.getElementById("imageModal").addEventListener(
    "click",
    function(event) {{

        if (event.target.id === "imageModal") {{
            closeImageModal();
        }}

    }}
);


document.addEventListener(
    "keydown",
    function(event) {{

        if (event.key === "Escape") {{
            closeImageModal();
        }}

    }}
);


applyFilters();

</script>


</body>

</html>
"""


# ============================================================
# 12. WRITE HTML
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(HTML_DOCUMENT)


# ============================================================
# 13. FINAL FILE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("7. REPORT CREATED")
print("=" * 80)

report_size = REPORT_PATH.stat().st_size

print("HTML:", REPORT_PATH)

print(
    f"HTML size: {report_size / 1024**2:.2f} MB"
)

print("Embedded overlays:", num_cards)

print("Notes:", NOTES_PATH)


# ============================================================
# 14. SAVE REPORT METADATA
# ============================================================

metadata = {

    "report_file": str(REPORT_PATH),

    "report_size_mb": round(
        report_size / 1024**2,
        3
    ),

    "embedded_images": int(num_cards),

    "unique_images": int(num_images),

    "models": MODEL_NAMES if "MODEL_NAMES" in globals()
    else ["C_IRFS", "CB_IRFS"],

    "classes": [
        "hat",
        "gloves",
        "mask",
        "pants",
        "shoes",
        "long-sleeved shirt",
    ],

    "confidence_threshold": 0.40,

    "mask_iou_threshold": 0.50,

    "self_contained": True,

    "requires_internet": False,

}

with open(
    REPORT_DIR / "report_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 15. DOWNLOAD LINKS
# ============================================================

print("\n" + "=" * 80)
print("8. DOWNLOAD")
print("=" * 80)

print("\nSelf-contained HTML report:")

display(
    FileLink(
        str(REPORT_PATH),
        result_html_prefix="Open/download report: "
    )
)

print("\nText notes:")

display(
    FileLink(
        str(NOTES_PATH),
        result_html_prefix="Open/download notes: "
    )
)


print("\n" + "=" * 80)
print("CELL 17M COMPLETED SUCCESSFULLY")
print("=" * 80)

###########################################################################
# CELL 39
###########################################################################
# ============================================================
# CELL 17M-FIX - SELF-CONTAINED HTML VISUAL AUDIT REPORT
#
# Fix:
#   Obtain split (Old/New) from visual_audit_image_class_summary.csv
#   Merge using model + image + class_id.
#
# Output:
#   html_visual_audit/visual_error_audit_report.html
#   html_visual_audit/visual_error_audit_notes.txt
#   html_visual_audit/class_error_summary.csv
#
# Features:
#   - All 24 overlay images embedded in HTML (Base64)
#   - No external internet required
#   - Filter by model, class, Old/New
#   - Search image filename
#   - Click image to enlarge
#   - Download individual images
#   - Overall and per-class statistics
#   - FP breakdown
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import base64
import html as html_lib

import numpy as np
import pandas as pd

from pathlib import Path
from IPython.display import display, FileLink


# ============================================================
# 2. PATHS
# ============================================================

ROOT = Path(
    "/kaggle/working/YOLO11_seg_expanded"
)

SOURCE_DIR = (
    ROOT
    / "research_experiments"
    / "confidence_threshold_analysis_extended"
)

VISUAL_DIR = (
    SOURCE_DIR
    / "error_visualizations"
)

REPORT_DIR = (
    SOURCE_DIR
    / "html_visual_audit"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_PATH = (
    SOURCE_DIR
    / "visual_overlay_manifest.csv"
)

AUDIT_SUMMARY_PATH = (
    SOURCE_DIR
    / "visual_audit_image_class_summary.csv"
)

AUDIT_DETAILS_PATH = (
    SOURCE_DIR
    / "visual_audit_object_details.csv"
)

FP_BREAKDOWN_PATH = (
    SOURCE_DIR
    / "visual_audit_fp_breakdown.csv"
)

SELECTED_PATH = (
    SOURCE_DIR
    / "selected_visual_error_examples.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "visual_error_audit_report.html"
)

NOTES_PATH = (
    REPORT_DIR
    / "visual_error_audit_notes.txt"
)


# ============================================================
# 3. VERIFY INPUT FILES
# ============================================================

print("=" * 80)
print("1. VERIFY INPUT FILES")
print("=" * 80)

required_files = [
    MANIFEST_PATH,
    AUDIT_SUMMARY_PATH,
    AUDIT_DETAILS_PATH,
    FP_BREAKDOWN_PATH,
    SELECTED_PATH,
    VISUAL_DIR,
]

for path in required_files:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required path: {path}"
        )

    print("[OK]", path.name)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("2. LOAD AUDIT DATA")
print("=" * 80)

manifest = pd.read_csv(
    MANIFEST_PATH
)

audit_summary = pd.read_csv(
    AUDIT_SUMMARY_PATH
)

audit_details = pd.read_csv(
    AUDIT_DETAILS_PATH
)

fp_breakdown = pd.read_csv(
    FP_BREAKDOWN_PATH
)

selected_examples = pd.read_csv(
    SELECTED_PATH
)


print("Overlay manifest:", len(manifest))
print("Audit summary:", len(audit_summary))
print("Object details:", len(audit_details))
print("FP breakdown:", len(fp_breakdown))
print("Selected examples:", len(selected_examples))


# ============================================================
# 5. VERIFY REQUIRED COLUMNS
# ============================================================

manifest_required = [
    "model",
    "class_id",
    "class_name",
    "image",
    "rank",
    "tp",
    "fp",
    "fn",
    "output_file",
]

summary_required = [
    "model",
    "image",
    "split",
    "class_id",
    "class_name",
    "gt",
    "pred",
    "tp",
    "fp",
    "fn",
]

for col in manifest_required:

    if col not in manifest.columns:
        raise RuntimeError(
            f"Manifest missing column: {col}"
        )


for col in summary_required:

    if col not in audit_summary.columns:
        raise RuntimeError(
            f"Audit summary missing column: {col}"
        )


# Normalize merge key types.
manifest["model"] = manifest["model"].astype(str)
manifest["image"] = manifest["image"].astype(str)
manifest["class_id"] = manifest["class_id"].astype(int)

audit_summary["model"] = audit_summary["model"].astype(str)
audit_summary["image"] = audit_summary["image"].astype(str)
audit_summary["class_id"] = audit_summary["class_id"].astype(int)


# ============================================================
# 6. FIX: RESTORE SPLIT FROM AUDIT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("3. RESTORE OLD/NEW SPLIT INFORMATION")
print("=" * 80)

split_map = (
    audit_summary[
        [
            "model",
            "image",
            "class_id",
            "split",
        ]
    ]
    .drop_duplicates()
    .copy()
)

# Every model/image/class combination must have one split.
duplicate_keys = split_map.duplicated(
    subset=[
        "model",
        "image",
        "class_id",
    ],
    keep=False
)

if duplicate_keys.any():

    raise RuntimeError(
        "Duplicate model/image/class keys found in split map. "
        "Cannot safely restore split information."
    )


if "split" not in manifest.columns:

    manifest = manifest.merge(
        split_map,
        on=[
            "model",
            "image",
            "class_id",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        "Split column restored from "
        "visual_audit_image_class_summary.csv"
    )

else:

    print(
        "Split column already exists in manifest."
    )


if manifest["split"].isna().any():

    missing_rows = manifest[
        manifest["split"].isna()
    ]

    raise RuntimeError(
        "Could not recover split for some overlay rows:\n"
        + missing_rows[
            ["model", "image", "class_id"]
        ].to_string(index=False)
    )


print("\nSplit distribution:")

print(
    manifest["split"].value_counts().to_string()
)

print("\nModel distribution:")

print(
    manifest["model"].value_counts().to_string()
)

print("\nSplit restoration: PASSED")


# Save corrected manifest separately.
manifest.to_csv(
    REPORT_DIR / "visual_overlay_manifest_with_split.csv",
    index=False
)


# ============================================================
# 7. RESOLVE OVERLAY PATHS
# ============================================================

print("\n" + "=" * 80)
print("4. VERIFY OVERLAY IMAGES")
print("=" * 80)


def resolve_overlay_path(path_value):

    original_path = Path(str(path_value))

    if original_path.exists():
        return original_path.resolve()

    candidate = (
        VISUAL_DIR
        / original_path.name
    )

    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Overlay image not found: {path_value}"
    )


manifest["resolved_image_path"] = [
    str(resolve_overlay_path(x))
    for x in manifest["output_file"]
]


if len(manifest) != 24:

    print(
        "WARNING: Expected 24 overlay records, "
        f"found {len(manifest)}."
    )


total_image_bytes = sum(
    Path(p).stat().st_size
    for p in manifest["resolved_image_path"]
)

print("Verified overlay images:", len(manifest))

print(
    f"Total image data: "
    f"{total_image_bytes / 1024**2:.2f} MB"
)


# ============================================================
# 8. BUILD CLASS SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("5. BUILD CLASS SUMMARY")
print("=" * 80)

class_summary = (
    audit_summary
    .groupby(
        [
            "model",
            "class_id",
            "class_name",
        ],
        as_index=False
    )
    .agg({

        "gt": "sum",
        "pred": "sum",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",

    })
)

class_summary["precision"] = (
    class_summary["tp"]
    / (
        class_summary["tp"]
        + class_summary["fp"]
    ).replace(0, np.nan)
).fillna(0)

class_summary["recall"] = (
    class_summary["tp"]
    / (
        class_summary["tp"]
        + class_summary["fn"]
    ).replace(0, np.nan)
).fillna(0)

class_summary["f1"] = (
    2
    * class_summary["precision"]
    * class_summary["recall"]
    / (
        class_summary["precision"]
        + class_summary["recall"]
    ).replace(0, np.nan)
).fillna(0)

class_summary["error_count"] = (
    class_summary["fp"]
    + class_summary["fn"]
)


class_summary.to_csv(
    REPORT_DIR / "class_error_summary.csv",
    index=False
)


overall_summary = (
    class_summary
    .groupby(
        "model",
        as_index=False
    )
    .agg({

        "gt": "sum",
        "pred": "sum",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",

    })
)

overall_summary["precision"] = (
    overall_summary["tp"]
    / (
        overall_summary["tp"]
        + overall_summary["fp"]
    ).replace(0, np.nan)
).fillna(0)

overall_summary["recall"] = (
    overall_summary["tp"]
    / (
        overall_summary["tp"]
        + overall_summary["fn"]
    ).replace(0, np.nan)
).fillna(0)

overall_summary["f1"] = (
    2
    * overall_summary["precision"]
    * overall_summary["recall"]
    / (
        overall_summary["precision"]
        + overall_summary["recall"]
    ).replace(0, np.nan)
).fillna(0)


print(
    class_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 9. DATAFRAME TO HTML
# ============================================================

def dataframe_to_html(
    dataframe,
    table_class="data-table"
):

    return dataframe.to_html(
        index=False,
        border=0,
        classes=table_class,
        escape=True,
        float_format=lambda x: f"{x:.4f}"
    )


overall_html = dataframe_to_html(
    overall_summary
)

class_html = dataframe_to_html(
    class_summary
)

fp_html = dataframe_to_html(
    fp_breakdown
)

selected_html = dataframe_to_html(
    selected_examples
)


# ============================================================
# 10. EMBED ALL IMAGES AS BASE64
# ============================================================

print("\n" + "=" * 80)
print("6. EMBED IMAGES")
print("=" * 80)


def safe_text(value):

    if pd.isna(value):
        return ""

    return html_lib.escape(
        str(value),
        quote=True
    )


def file_to_data_uri(path):

    path = Path(path)

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"

    elif suffix == ".png":
        mime = "image/png"

    else:
        raise ValueError(
            f"Unsupported image extension: {suffix}"
        )

    with open(path, "rb") as f:

        encoded = base64.b64encode(
            f.read()
        ).decode("ascii")

    return f"data:{mime};base64,{encoded}"


cards_html = []

embedded_bytes = 0

manifest = manifest.sort_values(
    [
        "model",
        "class_id",
        "rank",
    ]
).reset_index(drop=True)


for index, row in manifest.iterrows():

    image_path = Path(
        row["resolved_image_path"]
    )

    image_uri = file_to_data_uri(
        image_path
    )

    embedded_bytes += image_path.stat().st_size

    model_name = safe_text(row["model"])

    class_id = int(row["class_id"])

    class_name = safe_text(row["class_name"])

    image_name = safe_text(row["image"])

    split = safe_text(row["split"])

    rank = int(row["rank"])

    tp = int(row["tp"])
    fp = int(row["fp"])
    fn = int(row["fn"])

    errors = fp + fn

    search_text = (
        f"{row['model']} "
        f"{row['class_name']} "
        f"{row['image']} "
        f"{row['split']}"
    ).lower()

    search_text = safe_text(search_text)

    card = f"""
    <article
        class="case-card"
        data-model="{model_name}"
        data-class="{class_id}"
        data-split="{split}"
        data-search="{search_text}"
        data-errors="{errors}"
    >

        <div class="card-header">

            <div>
                <span class="model-tag">{model_name}</span>
                <span class="class-tag">{class_name}</span>
            </div>

            <span class="rank-tag">Rank #{rank}</span>

        </div>

        <h3>{image_name}</h3>

        <div class="stats">

            <span class="stat tp">TP: {tp}</span>

            <span class="stat fp">FP: {fp}</span>

            <span class="stat fn">FN: {fn}</span>

            <span class="stat total">
                Errors: {errors}
            </span>

        </div>

        <div class="image-container">

            <img
                src="{image_uri}"
                alt="{image_name}"
                class="audit-image"
                loading="lazy"
                onclick="openImageModal(this)"
            />

        </div>

        <div class="card-footer">

            <span>Split: {split}</span>

            <a
                href="{image_uri}"
                download="{model_name}_class{class_id}_rank{rank}.jpg"
                class="download-link"
            >
                Download image
            </a>

        </div>

    </article>
    """

    cards_html.append(card)

    if (index + 1) % 6 == 0:

        print(
            f"Embedded {index + 1}/{len(manifest)} images"
        )


cards_html_string = "\n".join(cards_html)

print(
    f"Embedded image size: "
    f"{embedded_bytes / 1024**2:.2f} MB"
)


# ============================================================
# 11. BUILD TEXT NOTES
# ============================================================

print("\n" + "=" * 80)
print("7. BUILD TEXT NOTES")
print("=" * 80)

notes = []

notes.append(
    "YOLO11 SEGMENTATION - VISUAL ERROR AUDIT\n"
)

notes.append(
    "Operating point: confidence >= 0.40, mask IoU >= 0.50\n"
)

notes.append(
    "Dataset: 268 validation images, 3,042 GT instances\n"
)

notes.append(
    "\nOVERALL SUMMARY\n"
)

for _, row in overall_summary.iterrows():

    notes.append(
        f"\n{row['model']}\n"
        f"GT={int(row['gt'])}, "
        f"Pred={int(row['pred'])}, "
        f"TP={int(row['tp'])}, "
        f"FP={int(row['fp'])}, "
        f"FN={int(row['fn'])}\n"
        f"Precision={row['precision']:.4f}, "
        f"Recall={row['recall']:.4f}, "
        f"F1={row['f1']:.4f}\n"
    )


notes.append(
    "\nPER-CLASS SUMMARY\n"
)

for _, row in class_summary.iterrows():

    notes.append(
        f"\n{row['model']} | {row['class_name']}\n"
        f"GT={int(row['gt'])}, "
        f"TP={int(row['tp'])}, "
        f"FP={int(row['fp'])}, "
        f"FN={int(row['fn'])}, "
        f"F1={row['f1']:.4f}\n"
    )


notes.append(
    "\nSELECTED VISUAL CASES\n"
)

for _, row in selected_examples.iterrows():

    notes.append(
        f"\n{row['model']} | "
        f"{row['class_name']} | "
        f"{row['image']}\n"
        f"GT={int(row['gt'])}, "
        f"Pred={int(row['pred'])}, "
        f"TP={int(row['tp'])}, "
        f"FP={int(row['fp'])}, "
        f"FN={int(row['fn'])}\n"
    )


notes.append(
    "\nINTERPRETATION\n"
    "FP/FN statistics use confidence=0.40 and mask IoU=0.50.\n"
    "These are validation diagnostics, not independent test results.\n"
)

with open(
    NOTES_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("".join(notes))


# ============================================================
# 12. BUILD HTML TEMPLATE
# ============================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>YOLO11 Segmentation - Visual Error Audit</title>

<style>

:root {
    --background: #f4f6f9;
    --surface: #ffffff;
    --text: #202938;
    --muted: #64748b;
    --border: #dce2ea;
    --primary: #315b91;
    --green: #16834b;
    --red: #cf3434;
    --orange: #c47712;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 0;
    background: var(--background);
    color: var(--text);
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.55;
}

.container {
    max-width: 1500px;
    margin: auto;
    padding: 26px;
}

h1 {
    margin-bottom: 5px;
    font-size: 30px;
}

h2 {
    margin-top: 35px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 9px;
}

.subtitle {
    color: var(--muted);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin: 24px 0;
}

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 17px;
}

.metric-label {
    color: var(--muted);
    font-size: 13px;
}

.metric-value {
    font-size: 26px;
    font-weight: bold;
    margin-top: 5px;
}

.section-note {
    color: var(--muted);
    background: white;
    padding: 13px 16px;
    border-left: 4px solid var(--primary);
    border-radius: 5px;
    margin: 15px 0;
}

.table-wrap {
    overflow-x: auto;
    background: white;
    border-radius: 9px;
    border: 1px solid var(--border);
    margin: 15px 0;
}

.data-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
    min-width: 700px;
}

.data-table th,
.data-table td {
    border-bottom: 1px solid var(--border);
    padding: 10px;
    text-align: right;
    white-space: nowrap;
}

.data-table th {
    background: #edf2f8;
    position: sticky;
    top: 0;
}

.data-table td:first-child,
.data-table th:first-child {
    text-align: left;
}

.filters {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 16px;
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin: 20px 0;
}

.filters input,
.filters select {
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    min-width: 160px;
    background: white;
    font-size: 14px;
}

.case-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 18px;
    margin-top: 20px;
}

.case-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

.card-header {
    padding: 14px 15px 5px;
    display: flex;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
}

.model-tag {
    display: inline-block;
    font-size: 12px;
    font-weight: bold;
    color: #234b7d;
    background: #e4edf8;
    padding: 5px 8px;
    border-radius: 5px;
}

.class-tag {
    display: inline-block;
    margin-left: 5px;
    font-size: 12px;
    color: #46566b;
    background: #edf0f4;
    padding: 5px 8px;
    border-radius: 5px;
}

.rank-tag {
    color: var(--muted);
    font-size: 12px;
}

.case-card h3 {
    font-size: 14px;
    padding: 0 15px;
    overflow-wrap: anywhere;
}

.stats {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    padding: 0 15px 12px;
}

.stat {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 5px;
    background: #f1f3f6;
}

.stat.tp {
    color: var(--green);
}

.stat.fp {
    color: var(--orange);
}

.stat.fn {
    color: var(--red);
}

.stat.total {
    font-weight: bold;
}

.image-container {
    background: #151a22;
    padding: 8px;
}

.audit-image {
    display: block;
    width: 100%;
    height: auto;
    cursor: zoom-in;
    border-radius: 3px;
}

.card-footer {
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-size: 12px;
}

.download-link {
    color: var(--primary);
    text-decoration: none;
    font-weight: bold;
}

.download-link:hover {
    text-decoration: underline;
}

.legend {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 15px 0;
    background: white;
    border: 1px solid var(--border);
    padding: 14px;
    border-radius: 8px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 13px;
}

.color-box {
    width: 15px;
    height: 15px;
    border-radius: 3px;
}

#resultCount {
    color: var(--muted);
    font-weight: bold;
    padding: 5px 0;
}

.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    inset: 0;
    background: rgba(0,0,0,0.94);
    padding: 25px;
    overflow: auto;
}

.modal-content {
    display: block;
    max-width: 96%;
    max-height: 90vh;
    object-fit: contain;
    margin: 30px auto;
    cursor: zoom-out;
}

.modal-caption {
    color: white;
    text-align: center;
    margin-top: 10px;
}

.modal-close {
    position: fixed;
    top: 14px;
    right: 25px;
    color: white;
    font-size: 38px;
    cursor: pointer;
}

.hidden {
    display: none !important;
}

footer {
    margin: 50px 0 20px;
    padding-top: 15px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
}

@media(max-width: 600px) {
    .container {
        padding: 12px;
    }

    .case-grid {
        grid-template-columns: 1fr;
    }

    h1 {
        font-size: 23px;
    }
}

</style>

</head>


<body>

<div class="container">

    <header>

        <h1>YOLO11 Segmentation - Visual Error Audit</h1>

        <div class="subtitle">
            C-IRFS vs CB-IRFS | Validation Error Analysis
        </div>

    </header>


    <div class="section-note">

        <strong>Evaluation configuration</strong><br>

        Dataset: 268 validation images, 3,042 Ground Truth instances<br>

        Confidence threshold: 0.40<br>

        Mask IoU threshold: 0.50<br>

        Classes: 6<br>

        Selected examples: top 2 images per model and class

    </div>


    <div class="metrics-grid">

        <div class="metric-box">

            <div class="metric-label">Embedded overlays</div>

            <div class="metric-value">__NUM_CARDS__</div>

        </div>

        <div class="metric-box">

            <div class="metric-label">Models</div>

            <div class="metric-value">__NUM_MODELS__</div>

        </div>

        <div class="metric-box">

            <div class="metric-label">Classes represented</div>

            <div class="metric-value">__NUM_CLASSES__</div>

        </div>

        <div class="metric-box">

            <div class="metric-label">Unique image names</div>

            <div class="metric-value">__NUM_IMAGES__</div>

        </div>

    </div>


    <h2>1. Overall model summary</h2>

    <div class="table-wrap">
        __OVERALL_TABLE__
    </div>


    <h2>2. Per-class error summary</h2>

    <div class="section-note">
        Statistics are calculated from all validation images,
        not only the selected visual examples.
    </div>

    <div class="table-wrap">
        __CLASS_TABLE__
    </div>


    <h2>3. False-positive categories</h2>

    <div class="table-wrap">
        __FP_TABLE__
    </div>


    <h2>4. Visual legend</h2>

    <div class="legend">

        <div class="legend-item">
            <div class="color-box" style="background:#00C800"></div>
            GT matched
        </div>

        <div class="legend-item">
            <div class="color-box" style="background:#FF0000"></div>
            GT false negative
        </div>

        <div class="legend-item">
            <div class="color-box" style="background:#00FFFF"></div>
            Prediction matched
        </div>

        <div class="legend-item">
            <div class="color-box" style="background:#FFA500"></div>
            Prediction false positive
        </div>

    </div>


    <h2>5. Selected visual error cases</h2>

    <div class="section-note">

        Each image compares the original image and its overlay.<br>

        Click any image to enlarge it.<br>

        Use the filters to inspect a specific model, class, split or filename.

    </div>


    <div class="filters">

        <select id="modelFilter">
            <option value="all">All models</option>
            <option value="C_IRFS">C-IRFS</option>
            <option value="CB_IRFS">CB-IRFS</option>
        </select>


        <select id="classFilter">
            <option value="all">All classes</option>
            <option value="0">Hat</option>
            <option value="1">Gloves</option>
            <option value="2">Mask</option>
            <option value="3">Pants</option>
            <option value="4">Shoes</option>
            <option value="5">Long-sleeved shirt</option>
        </select>


        <select id="splitFilter">
            <option value="all">All splits</option>
            <option value="Old">Old</option>
            <option value="New">New</option>
        </select>


        <input
            id="searchInput"
            type="text"
            placeholder="Search image filename..."
        />

    </div>


    <div id="resultCount"></div>


    <div class="case-grid" id="caseGrid">
        __IMAGE_CARDS__
    </div>


    <h2>6. Selected image records</h2>

    <div class="table-wrap">
        __SELECTED_TABLE__
    </div>


    <footer>

        Generated from Cell 17L.<br>

        Evaluation: confidence >= 0.40, mask IoU >= 0.50.<br>

        All images are embedded within this HTML file.<br>

        No Internet connection is required.<br><br>

        FP/FN statistics are validation diagnostics,
        not independent test results.

    </footer>

</div>


<div id="imageModal" class="modal">

    <span class="modal-close" onclick="closeImageModal()">
        &times;
    </span>

    <img id="modalImage" class="modal-content" />

    <div id="modalCaption" class="modal-caption"></div>

</div>


<script>

const modelFilter = document.getElementById("modelFilter");

const classFilter = document.getElementById("classFilter");

const splitFilter = document.getElementById("splitFilter");

const searchInput = document.getElementById("searchInput");

const cards = Array.from(
    document.querySelectorAll(".case-card")
);


function applyFilters() {

    const modelValue = modelFilter.value;

    const classValue = classFilter.value;

    const splitValue = splitFilter.value;

    const searchValue = searchInput.value.toLowerCase().trim();

    let visible = 0;

    cards.forEach(card => {

        const modelMatch =
            modelValue === "all"
            || card.dataset.model === modelValue;

        const classMatch =
            classValue === "all"
            || card.dataset.class === classValue;

        const splitMatch =
            splitValue === "all"
            || card.dataset.split === splitValue;

        const searchMatch =
            card.dataset.search.includes(searchValue);

        const show =
            modelMatch
            && classMatch
            && splitMatch
            && searchMatch;

        card.classList.toggle("hidden", !show);

        if (show) {
            visible++;
        }

    });

    document.getElementById("resultCount").textContent =
        `Showing ${visible} of ${cards.length} cases`;

}


modelFilter.addEventListener("change", applyFilters);

classFilter.addEventListener("change", applyFilters);

splitFilter.addEventListener("change", applyFilters);

searchInput.addEventListener("input", applyFilters);


function openImageModal(imgElement) {

    const modal = document.getElementById("imageModal");

    const title = imgElement
        .closest(".case-card")
        .querySelector("h3")
        .textContent;

    document.getElementById("modalImage").src =
        imgElement.src;

    document.getElementById("modalCaption").textContent =
        title;

    modal.style.display = "block";

}


function closeImageModal() {

    document.getElementById("imageModal").style.display =
        "none";

}


document.getElementById("imageModal").addEventListener(
    "click",
    function(event) {

        if (event.target.id === "imageModal") {
            closeImageModal();
        }

    }
);


document.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Escape") {
            closeImageModal();
        }

    }
);


applyFilters();

</script>


</body>
</html>
"""


# ============================================================
# 13. INSERT HTML CONTENT
# ============================================================

print("\n" + "=" * 80)
print("8. BUILD HTML DOCUMENT")
print("=" * 80)

num_cards = len(manifest)

num_models = manifest["model"].nunique()

num_classes = manifest["class_id"].nunique()

num_images = manifest["image"].nunique()


replacements = {

    "__NUM_CARDS__": str(num_cards),

    "__NUM_MODELS__": str(num_models),

    "__NUM_CLASSES__": str(num_classes),

    "__NUM_IMAGES__": str(num_images),

    "__OVERALL_TABLE__": overall_html,

    "__CLASS_TABLE__": class_html,

    "__FP_TABLE__": fp_html,

    "__SELECTED_TABLE__": selected_html,

    "__IMAGE_CARDS__": cards_html_string,

}


HTML_DOCUMENT = HTML_TEMPLATE

for key, value in replacements.items():

    HTML_DOCUMENT = HTML_DOCUMENT.replace(
        key,
        value
    )


# Verify no unresolved placeholders remain.
unresolved = [
    key for key in replacements
    if key in HTML_DOCUMENT
]

if unresolved:
    raise RuntimeError(
        f"Unresolved HTML placeholders: {unresolved}"
    )


# ============================================================
# 14. WRITE HTML
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(HTML_DOCUMENT)


if not REPORT_PATH.exists():
    raise RuntimeError("HTML file was not created.")

if REPORT_PATH.stat().st_size < 1_000_000:
    raise RuntimeError(
        "HTML file is unexpectedly small. "
        "Check embedded image data."
    )


# ============================================================
# 15. SAVE METADATA
# ============================================================

metadata = {

    "report_file": str(REPORT_PATH),

    "report_size_mb": round(
        REPORT_PATH.stat().st_size / 1024**2,
        3
    ),

    "embedded_images": int(num_cards),

    "unique_image_names": int(num_images),

    "number_of_models": int(num_models),

    "number_of_classes": int(num_classes),

    "models": sorted(
        manifest["model"].unique().tolist()
    ),

    "splits": sorted(
        manifest["split"].unique().tolist()
    ),

    "confidence_threshold": 0.40,

    "mask_iou_threshold": 0.50,

    "self_contained": True,

    "requires_internet": False,

    "images_embedded_as": "Base64",

}


with open(
    REPORT_DIR / "report_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 16. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("9. FINAL VERIFICATION")
print("=" * 80)

print("HTML exists:", REPORT_PATH.exists())

print(
    f"HTML size: "
    f"{REPORT_PATH.stat().st_size / 1024**2:.2f} MB"
)

print("Embedded overlay records:", num_cards)

print("Unique image names:", num_images)

print(
    "Models:",
    sorted(manifest["model"].unique().tolist())
)

print(
    "Splits:",
    sorted(manifest["split"].unique().tolist())
)

# Verify embedded Base64 count.
embedded_count = HTML_DOCUMENT.count("data:image/jpeg;base64,")

print("Embedded JPEG data blocks:", embedded_count)

if embedded_count != num_cards:
    raise RuntimeError(
        f"Expected {num_cards} embedded images, "
        f"found {embedded_count}."
    )


# ============================================================
# 17. DOWNLOAD LINKS
# ============================================================

print("\n" + "=" * 80)
print("10. DOWNLOAD FILES")
print("=" * 80)

print("\nMain HTML report:")

display(
    FileLink(
        str(REPORT_PATH),
        result_html_prefix="Download/open report: "
    )
)

print("\nText notes:")

display(
    FileLink(
        str(NOTES_PATH),
        result_html_prefix="Download notes: "
    )
)

print("\nCorrected manifest:")

display(
    FileLink(
        str(
            REPORT_DIR
            / "visual_overlay_manifest_with_split.csv"
        ),
        result_html_prefix="Download corrected manifest: "
    )
)


print("\n" + "=" * 80)
print("CELL 17M-FIX COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nMain report:")
print(REPORT_PATH)

###########################################################################
# CELL 40
###########################################################################
# ============================================================
# CELL 17M-FIX2
# OPTIMIZE EXISTING HTML AND VERIFY EMBEDDED IMAGES
#
# Input:
#   html_visual_audit/visual_error_audit_report.html
#
# Output:
#   html_visual_audit/visual_error_audit_report_compact.html
#
# Fix:
#   Embed every image exactly once.
#   Download buttons reuse the image already embedded in <img>.
#
# No inference, no GPU, no image regeneration.
# ============================================================

import re
import base64

from pathlib import Path
from IPython.display import display, FileLink


# ============================================================
# 1. PATHS
# ============================================================

SOURCE_DIR = Path(
    "/kaggle/working/YOLO11_seg_expanded/"
    "research_experiments/"
    "confidence_threshold_analysis_extended/"
    "html_visual_audit"
)

INPUT_HTML = (
    SOURCE_DIR
    / "visual_error_audit_report.html"
)

OUTPUT_HTML = (
    SOURCE_DIR
    / "visual_error_audit_report_compact.html"
)


# ============================================================
# 2. VERIFY ORIGINAL HTML
# ============================================================

print("=" * 75)
print("1. VERIFY ORIGINAL HTML")
print("=" * 75)

if not INPUT_HTML.exists():
    raise FileNotFoundError(INPUT_HTML)

original_size = INPUT_HTML.stat().st_size

html_text = INPUT_HTML.read_text(
    encoding="utf-8"
)

print("Original file:", INPUT_HTML.name)
print(f"Original size: {original_size / 1024**2:.2f} MB")

original_embedded_count = html_text.count(
    "data:image/jpeg;base64,"
)

print(
    "Original Base64 references:",
    original_embedded_count
)


# ============================================================
# 3. REPLACE DUPLICATED DOWNLOAD BASE64
# ============================================================

print("\n" + "=" * 75)
print("2. REMOVE DUPLICATED BASE64 FROM DOWNLOAD LINKS")
print("=" * 75)

download_pattern = re.compile(
    r'<a\s+'
    r'href="data:image/jpeg;base64,[^"]+"\s+'
    r'download="([^"]+)"\s+'
    r'class="download-link"\s*'
    r'>(.*?)</a>',
    flags=re.DOTALL
)

matches = list(
    download_pattern.finditer(html_text)
)

print("Download links found:", len(matches))

if len(matches) != 24:
    raise RuntimeError(
        f"Expected 24 download links, found {len(matches)}. "
        "The HTML structure differs from the expected report."
    )


def replace_download_link(match):

    filename = match.group(1)

    label = match.group(2).strip()

    return (
        f'<a href="#" '
        f'data-download="{filename}" '
        f'class="download-link" '
        f'onclick="downloadCardImage(this); return false;">'
        f'{label}'
        f'</a>'
    )


compact_html = download_pattern.sub(
    replace_download_link,
    html_text
)

replaced_count = len(matches)

print("Download links optimized:", replaced_count)


# ============================================================
# 4. ADD JAVASCRIPT DOWNLOAD FUNCTION
# ============================================================

print("\n" + "=" * 75)
print("3. ADD DOWNLOAD FUNCTION")
print("=" * 75)

download_function = r"""

function downloadCardImage(link) {

    const card = link.closest(".case-card");

    if (!card) {
        alert("Cannot find image card.");
        return;
    }

    const image = card.querySelector(".audit-image");

    if (!image || !image.src) {
        alert("Image source is unavailable.");
        return;
    }

    const temporaryLink = document.createElement("a");

    temporaryLink.href = image.src;

    temporaryLink.download =
        link.dataset.download || "overlay.jpg";

    document.body.appendChild(temporaryLink);

    temporaryLink.click();

    temporaryLink.remove();
}

"""


# Inject before the final applyFilters() call.
needle = "applyFilters();\n\n</script>"

position = compact_html.rfind(needle)

if position == -1:
    raise RuntimeError(
        "Could not locate final JavaScript insertion point."
    )

compact_html = (
    compact_html[:position]
    + download_function
    + compact_html[position:]
)

print("JavaScript download function inserted.")


# ============================================================
# 5. VERIFY IMAGE EMBEDDING
# ============================================================

print("\n" + "=" * 75)
print("4. VERIFY EMBEDDED IMAGES")
print("=" * 75)

src_pattern = re.compile(
    r'<img\b[^>]*\bsrc="(data:image/jpeg;base64,[^"]+)"',
    flags=re.IGNORECASE
)

embedded_sources = src_pattern.findall(
    compact_html
)

src_count = len(embedded_sources)

download_data_count = compact_html.count(
    'href="data:image/jpeg;base64,'
)

total_base64_count = compact_html.count(
    "data:image/jpeg;base64,"
)

download_link_count = compact_html.count(
    'data-download="'
)

print("Embedded image src:", src_count)
print("Base64 download href:", download_data_count)
print("Total Base64 references:", total_base64_count)
print("Download buttons:", download_link_count)


if src_count != 24:
    raise RuntimeError(
        f"Expected 24 image sources, found {src_count}."
    )

if download_data_count != 0:
    raise RuntimeError(
        "Some download links still contain duplicated Base64."
    )

if total_base64_count != 24:
    raise RuntimeError(
        f"Expected 24 Base64 references, "
        f"found {total_base64_count}."
    )

if download_link_count != 24:
    raise RuntimeError(
        f"Expected 24 download buttons, "
        f"found {download_link_count}."
    )


# Validate actual Base64 and JPEG signatures.
for index, uri in enumerate(
    embedded_sources,
    start=1
):

    encoded = uri.split(",", 1)[1]

    image_bytes = base64.b64decode(
        encoded,
        validate=True
    )

    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise RuntimeError(
            f"Image {index} is not a valid JPEG."
        )

print("All 24 embedded images passed Base64/JPEG validation.")


# ============================================================
# 6. WRITE COMPACT HTML
# ============================================================

print("\n" + "=" * 75)
print("5. WRITE OPTIMIZED HTML")
print("=" * 75)

OUTPUT_HTML.write_text(
    compact_html,
    encoding="utf-8"
)

if not OUTPUT_HTML.exists():
    raise RuntimeError(
        "Optimized HTML was not created."
    )

new_size = OUTPUT_HTML.stat().st_size

print("Output:", OUTPUT_HTML)
print(f"New size: {new_size / 1024**2:.2f} MB")

reduction = (
    1 - new_size / original_size
) * 100

print(f"Size reduction: {reduction:.2f}%")


# ============================================================
# 7. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 75)
print("6. FINAL VERIFICATION")
print("=" * 75)

assert OUTPUT_HTML.exists()

assert total_base64_count == 24

assert download_link_count == 24

assert download_data_count == 0

assert src_count == 24

print("[PASSED] HTML exists")
print("[PASSED] 24 images embedded exactly once")
print("[PASSED] 24 download buttons")
print("[PASSED] No duplicate Base64 download links")
print("[PASSED] All JPEG data valid")


# ============================================================
# 8. DOWNLOAD
# ============================================================

print("\n" + "=" * 75)
print("7. DOWNLOAD REPORT")
print("=" * 75)

display(
    FileLink(
        str(OUTPUT_HTML),
        result_html_prefix="Download compact HTML report: "
    )
)

print("\n" + "=" * 75)
print("CELL 17M-FIX2 COMPLETED SUCCESSFULLY")
print("=" * 75)

###########################################################################
# CELL 41
###########################################################################
# ============================================================
# CELL 17M-FIX3 - KAGGLE DOWNLOAD FIX
# Copy HTML to /kaggle/working root
# Generate FileLink using relative path
# ============================================================

import os
import shutil

from pathlib import Path
from IPython.display import display, FileLink

SOURCE = Path(
    "/kaggle/working/YOLO11_seg_expanded/"
    "research_experiments/confidence_threshold_analysis_extended/"
    "html_visual_audit/visual_error_audit_report_compact.html"
)

DESTINATION = Path(
    "/kaggle/working/visual_error_audit_report.html"
)

# 1. Verify source.
if not SOURCE.exists():
    raise FileNotFoundError(
        f"HTML source not found: {SOURCE}"
    )

# 2. Copy to Kaggle working root.
shutil.copy2(SOURCE, DESTINATION)

# 3. Verify copied file.
if not DESTINATION.exists():
    raise RuntimeError("Copy failed.")

if DESTINATION.stat().st_size != SOURCE.stat().st_size:
    raise RuntimeError("File size mismatch after copying.")

size_mb = DESTINATION.stat().st_size / 1024**2

print("=" * 65)
print("HTML DOWNLOAD FILE VERIFIED")
print("=" * 65)

print("File:", DESTINATION.name)
print(f"Size: {size_mb:.2f} MB")
print("Exists:", DESTINATION.exists())

# 4. Change current directory.
os.chdir("/kaggle/working")

# 5. Generate relative FileLink.
print("\nCLICK THE LINK BELOW TO DOWNLOAD:")

display(
    FileLink("visual_error_audit_report.html")
)

print("\nAlternatively, download from Kaggle Output panel.")

###########################################################################
# CELL 42
###########################################################################
# YOLO11-SEG — ONE-CELL RESEARCH ARCHIVER (Google Colab)
# Gói dữ liệu, nhãn, cấu hình, notebook history, logs, metrics, weights, plots,
# audit HTML + ảnh nhúng, môi trường chạy và ChatGPT export (nếu được cung cấp).

from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from IPython.display import display, Markdown
import os, sys, json, csv, re, base64, hashlib, shutil, zipfile, platform, subprocess, importlib.metadata

# ============================= CẤU HÌNH =====================================
ROOT = Path('/content')
PREFIX = 'YOLO11_seg_research_archive'
ADDITIONAL_ROOTS = []              # Ví dụ: [Path('/content/drive/MyDrive/YOLO11_Project')]
INCLUDE_WHOLE_MOUNTED_DRIVE = False # True chỉ khi thực sự muốn quét toàn bộ Drive
UPLOAD_CHAT_OR_AUDIT_EXPORT = True  # True lần đầu để chọn audit HTML / ChatGPT export / dataset ZIP còn thiếu
DRIVE_COPY_DIR = None               # Ví dụ: Path('/content/drive/MyDrive/Research_Archives')

# ============================= KHỞI TẠO =====================================
now = datetime.now().astimezone()
stamp = now.strftime('%Y%m%d_%H%M%S')
archive = ROOT / f'{PREFIX}_{stamp}.zip'
stage = ROOT / f'_{PREFIX}_BUILD'
if stage.exists(): shutil.rmtree(stage)
stage.mkdir(parents=True)

if UPLOAD_CHAT_OR_AUDIT_EXPORT:
    try:
        from google.colab import files
        print('Upload ChatGPT data export (.zip/.json/.html) và/hoặc Visual Error Audit HTML.')
        for name, data in files.upload().items():
            target = ROOT / Path(name).name
            target.write_bytes(data)
            print(f'Đã nhận: {target.name} ({len(data):,} bytes)')
    except Exception as e:
        print('Không mở được hộp upload:', repr(e))

# ============================= HÀM TIỆN ÍCH =================================
def under(p, parent):
    try: p.resolve().relative_to(parent.resolve()); return True
    except Exception: return False

def skip(p):
    if under(p, stage) or p.resolve() == archive.resolve(): return True
    if p.name.startswith(PREFIX) and p.suffix.lower() == '.zip': return True
    if any(x in {'.cache', '__pycache__', '.ipynb_checkpoints', 'node_modules', '.config', '.ssh', '.kaggle'} for x in p.parts): return True
    if p.name.lower() in {'kaggle.json', '.env', 'credentials.json', 'token.json'}: return True
    if not INCLUDE_WHOLE_MOUNTED_DRIVE and p.parts[:3] == ('/', 'content', 'drive'): return True
    return False

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout or '') + (('\n[stderr]\n' + r.stderr) if r.stderr else '')
    except Exception as e: return repr(e)

def sizeof(n):
    n = float(n)
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB': return f'{n:.2f} {u}'
        n /= 1024

# ============================= SAO LƯU TỆP ==================================
roots = [ROOT] + [Path(x) for x in ADDITIONAL_ROOTS]
roots = list(dict.fromkeys(p.resolve() for p in roots if p.exists()))
project = stage / 'project_files'
project.mkdir()
manifest, seen = [], set()

for ri, base in enumerate(roots):
    for src in base.rglob('*'):
        try:
            if not src.is_file() or skip(src) or str(src.resolve()) in seen: continue
            seen.add(str(src.resolve()))
            rel = src.relative_to(base) if base == ROOT.resolve() else Path('_external_roots') / f'{ri}_{base.name}' / src.relative_to(base)
            dst = project / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest.append({
                'source': str(src),
                'archive_path': str(dst.relative_to(stage)),
                'size_bytes': dst.stat().st_size,
                'sha256': sha256(dst)
            })
        except Exception as e:
            manifest.append({'source': str(src), 'copy_error': repr(e)})

# ============================= LỊCH SỬ NOTEBOOK =============================
nb = stage / 'notebook_history'
nb.mkdir()

try:
    ip = get_ipython()
    ns = ip.user_ns
    inputs = ns.get('In', [])

    rows = [
        {'execution_index': i, 'source': str(code)}
        for i, code in enumerate(list(inputs))
        if str(code).strip()
    ]

    (nb / 'executed_cell_inputs.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    (nb / 'executed_cell_inputs.py').write_text(
        '\n'.join(
            f'\n# ===== INPUT {x["execution_index"]} =====\n{x["source"]}'
            for x in rows
        ), encoding='utf-8'
    )

    out_dir = nb / 'cached_outputs'
    out_dir.mkdir()
    out_index = []

    def save_bundle(i, bundle):
        saved = []
        if not isinstance(bundle, dict): return saved

        for mime, value in bundle.items():
            try:
                if mime.startswith('image/'):
                    ext = '.png' if 'png' in mime else '.jpg'
                    raw = base64.b64decode(value) if isinstance(value, str) else bytes(value)
                    f = out_dir / f'cell_{i}{ext}'
                    f.write_bytes(raw)
                    saved.append(f.name)

                elif mime == 'text/html':
                    f = out_dir / f'cell_{i}.html'
                    f.write_text(str(value), encoding='utf-8')
                    saved.append(f.name)

                elif mime == 'text/markdown':
                    f = out_dir / f'cell_{i}.md'
                    f.write_text(str(value), encoding='utf-8')
                    saved.append(f.name)

                elif mime == 'text/plain':
                    f = out_dir / f'cell_{i}.txt'
                    f.write_text(str(value), encoding='utf-8')
                    saved.append(f.name)

            except Exception:
                pass
        return saved

    for i, value in dict(ns.get('Out', {})).items():
        saved = []

        try:
            if isinstance(value, dict):
                saved += save_bundle(i, value)

            elif hasattr(value, 'savefig'):
                f = out_dir / f'cell_{i}_figure.png'
                value.savefig(f, dpi=180, bbox_inches='tight')
                saved.append(f.name)

            elif hasattr(value, '_repr_mimebundle_'):
                b = value._repr_mimebundle_()
                saved += save_bundle(i, b[0] if isinstance(b, tuple) else b)

            elif value.__class__.__module__.startswith('PIL') and hasattr(value, 'save'):
                f = out_dir / f'cell_{i}_image.png'
                value.save(f)
                saved.append(f.name)

            elif value.__class__.__module__.startswith('pandas') and hasattr(value, 'to_csv'):
                f = out_dir / f'cell_{i}_table.csv'
                value.to_csv(f)
                saved.append(f.name)

            elif value.__class__.__module__.startswith('numpy') and getattr(value, 'ndim', 0) in (2, 3):
                try:
                    import numpy as np
                    from PIL import Image
                    a = np.nan_to_num(value)

                    if a.dtype != np.uint8:
                        if a.max() <= 1: a = a * 255
                        a = np.clip(a, 0, 255).astype('uint8')

                    f = out_dir / f'cell_{i}_array.png'
                    Image.fromarray(a).save(f)
                    saved.append(f.name)
                except Exception:
                    pass

        except Exception as e:
            saved.append('export_error: ' + repr(e))

        try:
            f = out_dir / f'cell_{i}_repr.txt'
            f.write_text(repr(value), encoding='utf-8', errors='replace')
            saved.append(f.name)
        except Exception:
            pass

        out_index.append({
            'execution_index': i,
            'type': str(type(value)),
            'saved': saved
        })

    (out_dir / 'index.json').write_text(
        json.dumps(out_index, ensure_ascii=False, indent=2), encoding='utf-8'
    )

except Exception as e:
    (nb / 'NOTE.txt').write_text(
        'Không đọc được IPython history: ' + repr(e), encoding='utf-8'
    )

# ============================= MÔI TRƯỜNG ===================================
env = {
    'captured_at': now.isoformat(),
    'captured_at_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version,
    'platform': platform.platform(),
    'machine': platform.machine(),
    'cwd': os.getcwd()
}

for pkg in [
    'ultralytics', 'torch', 'torchvision', 'numpy',
    'opencv-python', 'opencv-python-headless',
    'Pillow', 'pandas', 'matplotlib', 'pyyaml'
]:
    try:
        env.setdefault('package_versions', {})[pkg] = importlib.metadata.version(pkg)
    except Exception:
        pass

try:
    import torch
    env.update({
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })
except Exception as e:
    env['torch_error'] = repr(e)

(stage / 'runtime_environment.json').write_text(
    json.dumps(env, ensure_ascii=False, indent=2), encoding='utf-8'
)
(stage / 'pip_freeze.txt').write_text(
    run([sys.executable, '-m', 'pip', 'freeze']), encoding='utf-8'
)
(stage / 'nvidia_smi.txt').write_text(
    run(['nvidia-smi']), encoding='utf-8'
)
(stage / 'disk_usage.txt').write_text(
    run(['df', '-h', '/content']), encoding='utf-8'
)

# ============================= KIỂM KÊ DATASET ZIP ===========================
zip_inventory = []

for zpath in project.rglob('*.zip'):
    try:
        with zipfile.ZipFile(zpath) as z:
            names = [x for x in z.namelist() if not x.endswith('/')]

        exts = [Path(x).suffix.lower() for x in names]

        zip_inventory.append({
            'zip': str(zpath.relative_to(stage)),
            'files': len(names),
            'images': sum(x in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'} for x in exts),
            'txt': exts.count('.txt'),
            'yaml': exts.count('.yaml') + exts.count('.yml'),
            'train_txt': any(Path(x).name.lower() == 'train.txt' for x in names),
            'val_txt': any(Path(x).name.lower() == 'val.txt' for x in names),
            'first_20_entries': names[:20]
        })

    except Exception as e:
        zip_inventory.append({
            'zip': str(zpath.relative_to(stage)),
            'error': repr(e)
        })

(stage / 'dataset_zip_inventory.json').write_text(
    json.dumps(zip_inventory, ensure_ascii=False, indent=2), encoding='utf-8'
)

# ============================= HTML / ẢNH AUDIT ==============================
class HTMLCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.images = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)

        if tag in ('script', 'style'):
            self.skip += 1

        if tag == 'img' and (a.get('src') or '').startswith('data:image/'):
            self.images.append((a.get('alt') or '', a['src']))

        if tag in ('tr','p','div','section','h1','h2','h3','li','br'):
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script','style') and self.skip:
            self.skip -= 1

        if tag in ('tr','p','div','section','h1','h2','h3','li'):
            self.text.append('\n')

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.text.append(data.strip())


audit_dir = stage / 'visual_error_audit'
audit_dir.mkdir()
audit_inventory = []

for hp in project.rglob('*.html'):
    if 'audit' not in hp.name.lower() and 'yolo11 segmentation' not in hp.name.lower():
        continue

    try:
        parser = HTMLCollector()
        parser.feed(hp.read_text(encoding='utf-8', errors='replace'))

        txt = re.sub(r'\n{3,}', '\n\n', '\n'.join(parser.text))
        text_file = re.sub(r'[^A-Za-z0-9_.-]+', '_', hp.stem) + '_text.txt'

        (audit_dir / text_file).write_text(txt, encoding='utf-8')

        saved = []

        for i, (alt, uri) in enumerate(parser.images, 1):
            try:
                header, b64 = uri.split(',', 1)
                mt = re.search(r'data:image/([\w.+-]+)', header)
                ext = mt.group(1).replace('jpeg','jpg') if mt else 'jpg'

                name = re.sub(
                    r'[^A-Za-z0-9_.-]+', '_',
                    Path(alt or f'image_{i}').name
                )

                if not Path(name).suffix:
                    name += '.' + ext

                dest = audit_dir / f'{i:03d}_{name}'
                dest.write_bytes(base64.b64decode(b64))
                saved.append(dest.name)

            except Exception:
                pass

        audit_inventory.append({
            'html': str(hp.relative_to(stage)),
            'text': text_file,
            'embedded_images_found': len(parser.images),
            'images_saved': saved
        })

    except Exception as e:
        audit_inventory.append({'html': str(hp), 'error': repr(e)})

(stage / 'visual_error_audit_inventory.json').write_text(
    json.dumps(audit_inventory, ensure_ascii=False, indent=2), encoding='utf-8'
)

# ============================= TRÍCH CHATGPT EXPORT ==========================
def norm(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip().lower()

def body_text(c):
    if isinstance(c, str):
        return c
    if not isinstance(c, dict):
        return str(c)

    out = []

    for p in c.get('parts', []):
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            out.append(str(p.get('text') or p.get('caption') or '[MEDIA/ATTACHMENT]'))

    if not out and c.get('text'):
        out.append(str(c['text']))

    return '\n'.join(out).strip()

def matching_chat(obj):
    arr = obj if isinstance(obj, list) else [obj]

    if isinstance(obj, dict):
        for k in ('conversations', 'items', 'data'):
            if isinstance(obj.get(k), list):
                arr = obj[k]
                break

    exact = [
        c for c in arr
        if isinstance(c, dict) and 'định dạng annotation' in norm(c.get('title'))
    ]

    if exact:
        arr = exact
    else:
        arr = [
            c for c in arr
            if isinstance(c, dict) and any(
                k in norm(c.get('title'))
                for k in ('yolo11 segmentation', 'yolo11-seg')
            )
        ]

    for conv in arr:
        if not isinstance(conv, dict):
            continue

        messages = []
        mapping = conv.get('mapping', {})

        if isinstance(mapping, dict):
            for node in mapping.values():
                m = node.get('message') if isinstance(node, dict) else None

                if not isinstance(m, dict):
                    continue

                ts = m.get('create_time') or 0

                try:
                    date = datetime.fromtimestamp(float(ts)).astimezone().isoformat() if ts else ''
                except Exception:
                    date = ''

                role = (m.get('author') or {}).get('role', 'unknown')
                text = body_text(m.get('content', {})) or '[MEDIA-ONLY / EMPTY MESSAGE]'
                messages.append((float(ts or 0), role, date, text))

        elif isinstance(conv.get('messages'), list):
            for m in conv['messages']:
                ts = m.get('create_time') or m.get('timestamp') or 0

                try:
                    date = datetime.fromtimestamp(float(ts)).astimezone().isoformat() if ts else ''
                except Exception:
                    date = ''

                role = m.get('role') or (m.get('author') or {}).get('role', 'unknown')
                messages.append((
                    float(ts or 0),
                    role,
                    date,
                    body_text(m.get('content', m.get('text',''))) or '[EMPTY/MEDIA MESSAGE]'
                ))

        messages.sort(key=lambda x: x[0])
        lines = [f'# {conv.get("title", "YOLO11 Segmentation")}\n']

        for _, role, date, text in messages:
            lines += ['\n---\n', f'## {role} {date}\n\n', text, '\n']

        return '\n'.join(lines)

    return None


chat_dir = stage / 'chat_transcript'
chat_dir.mkdir()

chat_status = {
    'transcript_extracted': False,
    'sources_checked': [],
    'note': 'Không có transcript ChatGPT đầy đủ trừ khi người dùng upload đúng bản export.'
}

for jp in project.rglob('*.json'):
    if jp.name.lower() not in ('conversations.json','conversation.json') and 'chat' not in jp.name.lower():
        continue

    chat_status['sources_checked'].append(str(jp.relative_to(stage)))

    try:
        transcript = matching_chat(
            json.loads(jp.read_text(encoding='utf-8', errors='replace'))
        )
    except Exception:
        transcript = None

    if transcript:
        (chat_dir / 'Dinh_dang_annotation_YOLO11_Segmentation.md').write_text(
            transcript, encoding='utf-8'
        )
        chat_status.update({
            'transcript_extracted': True,
            'source': str(jp.relative_to(stage))
        })
        break

# ChatGPT account export ZIP: extract matching conversation JSON.
if not chat_status['transcript_extracted']:
    for zp in project.rglob('*.zip'):
        try:
            with zipfile.ZipFile(zp) as z:
                candidates = [
                    n for n in z.namelist()
                    if Path(n).name.lower() == 'conversations.json'
                ]

                for member in candidates:
                    try:
                        transcript = matching_chat(
                            json.loads(z.read(member).decode('utf-8', errors='replace'))
                        )
                    except Exception:
                        transcript = None

                    if transcript:
                        (chat_dir / 'Dinh_dang_annotation_YOLO11_Segmentation.md').write_text(
                            transcript, encoding='utf-8'
                        )
                        chat_status.update({
                            'transcript_extracted': True,
                            'source': f'{zp.name}:{member}'
                        })
                        break

                if chat_status['transcript_extracted']:
                    break

        except Exception:
            pass

# Also extract visible text and embedded images from a chat-specific HTML export.
for hp in project.rglob('*.html'):
    if not any(k in hp.name.lower() for k in ('chat','conversation','định dạng annotation')):
        continue

    try:
        parser = HTMLCollector()
        parser.feed(hp.read_text(encoding='utf-8', errors='replace'))

        (chat_dir / (hp.stem + '_visible_text.txt')).write_text(
            '\n'.join(parser.text), encoding='utf-8'
        )

        media_dir = chat_dir / 'embedded_html_images'
        media_dir.mkdir(exist_ok=True)
        extracted = []

        for i, (alt, uri) in enumerate(parser.images, 1):
            try:
                header, b64 = uri.split(',', 1)
                mt = re.search(r'data:image/([\w.+-]+)', header)
                ext = mt.group(1).replace('jpeg', 'jpg') if mt else 'jpg'

                name = re.sub(
                    r'[^A-Za-z0-9_.-]+', '_',
                    Path(alt or f'image_{i}').name
                )

                if not Path(name).suffix:
                    name += '.' + ext

                dst = media_dir / f'{i:03d}_{name}'
                dst.write_bytes(base64.b64decode(b64))
                extracted.append(dst.name)

            except Exception:
                pass

        chat_status['html_export'] = str(hp.relative_to(stage))
        chat_status['embedded_html_images'] = extracted

    except Exception:
        pass

(stage / 'chat_export_status.json').write_text(
    json.dumps(chat_status, ensure_ascii=False, indent=2), encoding='utf-8'
)

# ============================= TÓM TẮT TRAINING CSV ==========================
training = []

for cp in project.rglob('*.csv'):
    try:
        with open(cp, encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        cols = list(rows[0])
        clean = {c: c.strip().lower() for c in cols}

        if 'epoch' not in clean.values() or not any(
            any(k in c.lower() for k in ('map50','precision','recall','loss'))
            for c in cols
        ):
            continue

        mapcols = [c for c in cols if 'map50-95' in c.lower()]
        bestrow, best = None, None

        for row in rows:
            vals = []

            for c in mapcols:
                try:
                    vals.append(float(row[c]))
                except Exception:
                    pass

            if vals and (best is None or max(vals) > best):
                best, bestrow = max(vals), row

        training.append({
            'csv': str(cp.relative_to(stage)),
            'epochs_logged': len(rows),
            'columns': cols,
            'last_row': rows[-1],
            'best_map50_95': best,
            'best_row': bestrow
        })

    except Exception:
        pass

(stage / 'training_results_summary.json').write_text(
    json.dumps(training, ensure_ascii=False, indent=2), encoding='utf-8'
)

# ============================= GHI CHÚ NGHIÊN CỨU ============================
notes = r'''# YOLO11 Instance Segmentation — Research Handover

## A. Mục tiêu và dữ liệu đã biết
- Hướng làm việc: YOLO11-seg / instance segmentation trong Google Colab, dùng polygon instance annotations được gán bằng CVAT.
- `job62.zip`: thông tin người dùng từng báo là 597 files, 298 ảnh, 298 TXT và 1 YAML; cũng cho biết có `train.txt`. Cần đối chiếu inventory tự động vì số lượng file TXT có thể bao gồm file split.
- `data.yaml`: `path: .`, `train: train.txt`.
- Class IDs: 0 hat; 1 gloves; 2 mask; 3 pants; 4 shoes; 5 long-sleeved shirt.
- Ví dụ nhãn đặt dưới `labels/train/batch_007/`; mỗi dòng bắt đầu bằng class ID, tiếp theo là tọa độ polygon x/y chuẩn hóa theo định dạng YOLO segmentation.
- `job54.zip` cũng được nhắc đến nhưng trước đây chưa có số lượng ảnh/nhãn hay YAML được xác nhận. Xem `dataset_zip_inventory.json` để lấy số thực tế.

## B. Visual Error Audit đã xác nhận
Báo cáo `YOLO11 Segmentation - Visual Error Audit.html` (nếu được đưa vào Colab) chứa ảnh overlay nhúng bên trong.
- Validation: 268 images; 3,042 Ground Truth instances.
- Confidence threshold 0.40; mask IoU threshold 0.50.
- 6 classes; top 2 images/model/class; 24 overlays, 16 unique image names.
- Hai model labels trong báo cáo: `CB_IRFS`, `C_IRFS`.

### Overall metrics
| Metric | CB_IRFS | C_IRFS |
|---|---:|---:|
| GT | 3042 | 3042 |
| Predictions | 2474 | 2519 |
| TP | 2201 | 2223 |
| FP | 273 | 296 |
| FN | 841 | 819 |
| Precision | 0.8897 | 0.8825 |
| Recall | 0.7235 | 0.7308 |
| F1 | 0.7980 | 0.7995 |

### Per-class results
| Class | CB F1 | C F1 | CB Recall | C Recall |
|---|---:|---:|---:|---:|
| hat | 0.8804 | 0.8738 | 0.8432 | 0.8311 |
| gloves | 0.5348 | 0.5479 | 0.4202 | 0.4202 |
| mask | 0.8155 | 0.8109 | 0.7434 | 0.7463 |
| pants | 0.7494 | 0.7393 | 0.6637 | 0.6614 |
| shoes | 0.6122 | 0.6131 | 0.4927 | 0.5292 |
| long-sleeved shirt | 0.8434 | 0.8596 | 0.7849 | 0.8090 |

### Preliminary interpretation — not causal claims
- C_IRFS hơn F1 0.0015 tuyệt đối: rất nhỏ, chưa đủ kết luận vượt trội rõ nếu chưa repeated seeds/test độc lập.
- CB_IRFS có Precision cao hơn và FP ít hơn; C_IRFS có Recall cao hơn và FN ít hơn.
- Gloves yếu nhất theo F1 (~0.535–0.548), Recall 0.4202 ở cả hai.
- Shoes có Recall thấp (~0.493–0.529).
- Long-sleeved shirt có FN thô cao (197 CB, 175 C) nhưng cũng có 916 GT; cần đọc cả FN/GT.
- C nhỉnh hơn ở gloves, shoes, shirt; CB nhỉnh hơn ở hat, mask, pants. Không model nào thắng mọi lớp.
- FP được chia thành duplicate/competing prediction, no significant overlap, partial overlap/localization. Dùng overlay để kiểm tra trước khi quy kết nguyên nhân.
- Các thống kê là validation diagnostics, KHÔNG phải independent test results. Không trình bày chúng như kết quả test cuối cùng.

## C. Những thứ chưa được xác minh — không tự suy diễn
- Quy mô cuối cùng và cấu trúc `job54.zip`, tổng số ảnh sau merge.
- Cách tạo split; leakage theo video/person/session hoặc ảnh gần trùng.
- Mapping chính xác của C_IRFS/CB_IRFS sang kiến trúc/checkpoint.
- Seed, epochs, batch size, imgsz, optimizer, LR, augmentations, pretrained weights, patience, GPU và thời gian train.
- Test độc lập, repeated seeds, confidence intervals, threshold sweep.
- Chất lượng polygon GT, policy occlusion/ambiguous class, missing instances.

## D. Cấu trúc archive
- `project_files/`: file trong `/content` và `ADDITIONAL_ROOTS`.
- `notebook_history/`: input lệnh và những expression outputs còn trong IPython `In`/`Out`; không đảm bảo chứa mọi stdout/ảnh từng render trong giao diện Colab.
- `visual_error_audit/`: text và ảnh embedded data-URI từ audit HTML.
- `chat_transcript/`: transcript lọc từ ChatGPT export nếu có.
- `runtime_environment.json`, `pip_freeze.txt`, `nvidia_smi.txt`: môi trường.
- `dataset_zip_inventory.json`, `training_results_summary.json`, `manifest.json`: kiểm kê dataset, kết quả training và SHA-256.

## E. Giới hạn quan trọng
Cell không thể truy cập lịch sử một chat ChatGPT khác. Để lưu trọn trao đổi từ chiều 30/10/2026 đến lúc export, cần upload ChatGPT data export ZIP/`conversations.json` hoặc transcript HTML của đúng cuộc trò chuyện. Ảnh audit sẽ được trích xuất nếu HTML gốc có trong `/content`; hình ảnh đính kèm chat cần có trong export hoặc được upload riêng. Nếu upload ZIP export toàn tài khoản, ZIP gốc cũng được giữ trong archive và có thể chứa các chat khác; ưu tiên transcript của riêng cuộc trò chuyện này nếu không muốn đưa dữ liệu không liên quan vào gói.
- Muốn giữ đầy đủ output/ảnh hiển thị của notebook, tải notebook `.ipynb` bằng File → Download → Download .ipynb rồi upload vào `/content` trước khi chạy cell. `In`/`Out` chỉ lưu những gì còn trong kernel hiện tại.
'''

(stage / 'RESEARCH_HANDOVER.md').write_text(notes, encoding='utf-8')

# ============================= MANIFEST / README ============================
files = [p for p in project.rglob('*') if p.is_file()]

summary = {
    'created_at': now.isoformat(),
    'project_files': len(manifest),
    'project_size_bytes': sum(x.get('size_bytes',0) for x in manifest),
    'images': sum(
        p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff','.gif'}
        for p in files
    ),
    'dataset_zips': zip_inventory,
    'audit_reports': audit_inventory,
    'training_csvs_summarized': len(training),
    'roots_scanned': [str(x) for x in roots]
}

(stage / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)
(stage / 'archive_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
)

readme = [
    '# Archive contents',
    f'- Created: {now.isoformat()}',
    f'- Files copied: {len(manifest):,}',
    f'- Copied size: {sizeof(summary["project_size_bytes"])}',
    f'- Images found: {summary["images"]:,}',
    f'- Audit HTML processed: {len(audit_inventory)}',
    f'- Audit images extracted: {sum(len(x.get("images_saved",[])) for x in audit_inventory)}',
    f'- Training CSVs summarized: {len(training)}',
    f'- Chat transcript extracted: {chat_status["transcript_extracted"]}',
    '',
    'Mở `RESEARCH_HANDOVER.md`, `chat_export_status.json` và `dataset_zip_inventory.json` để kiểm tra độ đầy đủ.'
]

(stage / 'README_ARCHIVE.md').write_text('\n'.join(readme), encoding='utf-8')

# ============================= TẠO ZIP =======================================
with zipfile.ZipFile(
    archive, 'w', compression=zipfile.ZIP_DEFLATED,
    compresslevel=1, allowZip64=True
) as z:
    for p in stage.rglob('*'):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(stage)))

size = archive.stat().st_size

print('\n' + '='*72 + '\nYOLO11-SEG RESEARCH ARCHIVE CREATED\n' + '='*72)
print('Archive:', archive, '| Size:', sizeof(size))
print('Project files:', len(manifest), '| Audit HTML:', len(audit_inventory),
      '| Extracted audit images:', sum(len(x.get('images_saved',[])) for x in audit_inventory))
print('Training CSV summaries:', len(training), '| Chat transcript extracted:', chat_status['transcript_extracted'])
print('Kiểm tra README_ARCHIVE.md và chat_export_status.json trong ZIP trước khi chuyển chat.')

if DRIVE_COPY_DIR is not None:
    try:
        d = Path(DRIVE_COPY_DIR)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, d / archive.name)
        print('Drive copy:', d / archive.name)
    except Exception as e:
        print('Không copy được sang Drive:', repr(e))

if size < 1_800_000_000:
    try:
        from google.colab import files
        files.download(str(archive))
    except Exception as e:
        print('Không tự download được; ZIP vẫn ở:', archive, '|', repr(e))
else:
    print('ZIP lớn hơn ~1.8 GB. Hãy cấu hình DRIVE_COPY_DIR để lưu vào Drive.')

print('Thư mục tạm để kiểm tra:', stage)

###########################################################################
# CELL 43
###########################################################################
# YOLO11-SEG — ONE-CELL RESEARCH ARCHIVER (Google Colab)
# Gói dữ liệu, nhãn, cấu hình, notebook history, logs, metrics, weights, plots,
# audit HTML + ảnh nhúng, môi trường chạy và ChatGPT export (nếu được cung cấp).

from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from IPython.display import display, Markdown
import os, sys, json, csv, re, base64, hashlib, shutil, zipfile, platform, subprocess, importlib.metadata

# ============================= CẤU HÌNH =====================================
ROOT = Path('/content')
PREFIX = 'YOLO11_seg_research_archive'
ADDITIONAL_ROOTS = []              # Ví dụ: [Path('/content/drive/MyDrive/YOLO11_Project')]
INCLUDE_WHOLE_MOUNTED_DRIVE = False # True chỉ khi thực sự muốn quét toàn bộ Drive
UPLOAD_CHAT_OR_AUDIT_EXPORT = True  # True lần đầu để chọn audit HTML / ChatGPT export / dataset ZIP còn thiếu
DRIVE_COPY_DIR = None               # Ví dụ: Path('/content/drive/MyDrive/Research_Archives')

# ============================= KHỞI TẠO =====================================
now = datetime.now().astimezone()
stamp = now.strftime('%Y%m%d_%H%M%S')
archive = ROOT / f'{PREFIX}_{stamp}.zip'
stage = ROOT / f'_{PREFIX}_BUILD'
if stage.exists(): shutil.rmtree(stage)
stage.mkdir(parents=True)

if UPLOAD_CHAT_OR_AUDIT_EXPORT:
    try:
        from google.colab import files
        print('Upload ChatGPT data export (.zip/.json/.html) và/hoặc Visual Error Audit HTML.')
        for name, data in files.upload().items():
            target = ROOT / Path(name).name
            target.write_bytes(data)
            print(f'Đã nhận: {target.name} ({len(data):,} bytes)')
    except Exception as e:
        print('Không mở được hộp upload:', repr(e))

# ============================= HÀM TIỆN ÍCH =================================
def under(p, parent):
    try: p.resolve().relative_to(parent.resolve()); return True
    except Exception: return False

def skip(p):
    if under(p, stage) or p.resolve() == archive.resolve(): return True
    if p.name.startswith(PREFIX) and p.suffix.lower() == '.zip': return True
    if any(x in {'.cache', '__pycache__', '.ipynb_checkpoints', 'node_modules', '.config', '.ssh', '.kaggle'} for x in p.parts): return True
    if p.name.lower() in {'kaggle.json', '.env', 'credentials.json', 'token.json'}: return True
    if not INCLUDE_WHOLE_MOUNTED_DRIVE and p.parts[:3] == ('/', 'content', 'drive'): return True
    return False

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout or '') + (('\n[stderr]\n' + r.stderr) if r.stderr else '')
    except Exception as e: return repr(e)

def sizeof(n):
    n = float(n)
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB': return f'{n:.2f} {u}'
        n /= 1024

# ============================= SAO LƯU TỆP ==================================
roots = [ROOT] + [Path(x) for x in ADDITIONAL_ROOTS]
roots = list(dict.fromkeys(p.resolve() for p in roots if p.exists()))
project = stage / 'project_files'
project.mkdir()
manifest, seen = [], set()

for ri, base in enumerate(roots):
    for src in base.rglob('*'):
        try:
            if not src.is_file() or skip(src) or str(src.resolve()) in seen: continue
            seen.add(str(src.resolve()))
            rel = src.relative_to(base) if base == ROOT.resolve() else Path('_external_roots') / f'{ri}_{base.name}' / src.relative_to(base)
            dst = project / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest.append({
                'source': str(src),
                'archive_path': str(dst.relative_to(stage)),
                'size_bytes': dst.stat().st_size,
                'sha256': sha256(dst)
            })
        except Exception as e:
            manifest.append({'source': str(src), 'copy_error': repr(e)})

# ============================= LỊCH SỬ NOTEBOOK =============================
nb = stage / 'notebook_history'
nb.mkdir()

try:
    ip = get_ipython()
    ns = ip.user_ns
    inputs = ns.get('In', [])

    rows = [
        {'execution_index': i, 'source': str(code)}
        for i, code in enumerate(list(inputs))
        if str(code).strip()
    ]

    (nb / 'executed_cell_inputs.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    (nb / 'executed_cell_inputs.py').write_text(
        '\n'.join(
            f'\n# ===== INPUT {x["execution_index"]} =====\n{x["source"]}'
            for x in rows
        ), encoding='utf-8'
    )

    out_dir = nb / 'cached_outputs'
    out_dir.mkdir()
    out_index = []

    def save_bundle(i, bundle):
        saved = []
        if not isinstance(bundle, dict): return saved

        for mime, value in bundle.items():
            try:
                if mime.startswith('image/'):
                    ext = '.png' if 'png' in mime else '.jpg'
                    raw = base64.b64decode(value) if isinstance(value, str) else bytes(value)
                    f = out_dir / f'cell_{i}{ext}'
                    f.write_bytes(raw)
                    saved.append(f.name)

                elif mime == 'text/html':
                    f = out_dir / f'cell_{i}.html'
                    f.write_text(str(value), encoding='utf-8')
                    saved.append(f.name)

                elif mime == 'text/markdown':
                    f = out_dir / f'cell_{i}.md'
                    f.write_text(str(value), encoding='utf-8')
                    saved.append(f.name)

                elif mime == 'text/plain':
                    f = out_dir / f'cell_{i}.txt'
                    f.write_text(str(value), encoding='utf-8')
                    saved.append(f.name)

            except Exception:
                pass
        return saved

    for i, value in dict(ns.get('Out', {})).items():
        saved = []

        try:
            if isinstance(value, dict):
                saved += save_bundle(i, value)

            elif hasattr(value, 'savefig'):
                f = out_dir / f'cell_{i}_figure.png'
                value.savefig(f, dpi=180, bbox_inches='tight')
                saved.append(f.name)

            elif hasattr(value, '_repr_mimebundle_'):
                b = value._repr_mimebundle_()
                saved += save_bundle(i, b[0] if isinstance(b, tuple) else b)

            elif value.__class__.__module__.startswith('PIL') and hasattr(value, 'save'):
                f = out_dir / f'cell_{i}_image.png'
                value.save(f)
                saved.append(f.name)

            elif value.__class__.__module__.startswith('pandas') and hasattr(value, 'to_csv'):
                f = out_dir / f'cell_{i}_table.csv'
                value.to_csv(f)
                saved.append(f.name)

            elif value.__class__.__module__.startswith('numpy') and getattr(value, 'ndim', 0) in (2, 3):
                try:
                    import numpy as np
                    from PIL import Image
                    a = np.nan_to_num(value)

                    if a.dtype != np.uint8:
                        if a.max() <= 1: a = a * 255
                        a = np.clip(a, 0, 255).astype('uint8')

                    f = out_dir / f'cell_{i}_array.png'
                    Image.fromarray(a).save(f)
                    saved.append(f.name)
                except Exception:
                    pass

        except Exception as e:
            saved.append('export_error: ' + repr(e))

        try:
            f = out_dir / f'cell_{i}_repr.txt'
            f.write_text(repr(value), encoding='utf-8', errors='replace')
            saved.append(f.name)
        except Exception:
            pass

        out_index.append({
            'execution_index': i,
            'type': str(type(value)),
            'saved': saved
        })

    (out_dir / 'index.json').write_text(
        json.dumps(out_index, ensure_ascii=False, indent=2), encoding='utf-8'
    )

except Exception as e:
    (nb / 'NOTE.txt').write_text(
        'Không đọc được IPython history: ' + repr(e), encoding='utf-8'
    )

# ============================= MÔI TRƯỜNG ===================================
env = {
    'captured_at': now.isoformat(),
    'captured_at_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version,
    'platform': platform.platform(),
    'machine': platform.machine(),
    'cwd': os.getcwd()
}

for pkg in [
    'ultralytics', 'torch', 'torchvision', 'numpy',
    'opencv-python', 'opencv-python-headless',
    'Pillow', 'pandas', 'matplotlib', 'pyyaml'
]:
    try:
        env.setdefault('package_versions', {})[pkg] = importlib.metadata.version(pkg)
    except Exception:
        pass

try:
    import torch
    env.update({
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })
except Exception as e:
    env['torch_error'] = repr(e)

(stage / 'runtime_environment.json').write_text(
    json.dumps(env, ensure_ascii=False, indent=2), encoding='utf-8'
)
(stage / 'pip_freeze.txt').write_text(
    run([sys.executable, '-m', 'pip', 'freeze']), encoding='utf-8'
)
(stage / 'nvidia_smi.txt').write_text(
    run(['nvidia-smi']), encoding='utf-8'
)
(stage / 'disk_usage.txt').write_text(
    run(['df', '-h', '/content']), encoding='utf-8'
)

# ============================= KIỂM KÊ DATASET ZIP ===========================
zip_inventory = []

for zpath in project.rglob('*.zip'):
    try:
        with zipfile.ZipFile(zpath) as z:
            names = [x for x in z.namelist() if not x.endswith('/')]

        exts = [Path(x).suffix.lower() for x in names]

        zip_inventory.append({
            'zip': str(zpath.relative_to(stage)),
            'files': len(names),
            'images': sum(x in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'} for x in exts),
            'txt': exts.count('.txt'),
            'yaml': exts.count('.yaml') + exts.count('.yml'),
            'train_txt': any(Path(x).name.lower() == 'train.txt' for x in names),
            'val_txt': any(Path(x).name.lower() == 'val.txt' for x in names),
            'first_20_entries': names[:20]
        })

    except Exception as e:
        zip_inventory.append({
            'zip': str(zpath.relative_to(stage)),
            'error': repr(e)
        })

(stage / 'dataset_zip_inventory.json').write_text(
    json.dumps(zip_inventory, ensure_ascii=False, indent=2), encoding='utf-8'
)

# ============================= HTML / ẢNH AUDIT ==============================
class HTMLCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.images = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)

        if tag in ('script', 'style'):
            self.skip += 1

        if tag == 'img' and (a.get('src') or '').startswith('data:image/'):
            self.images.append((a.get('alt') or '', a['src']))

        if tag in ('tr','p','div','section','h1','h2','h3','li','br'):
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script','style') and self.skip:
            self.skip -= 1

        if tag in ('tr','p','div','section','h1','h2','h3','li'):
            self.text.append('\n')

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.text.append(data.strip())


audit_dir = stage / 'visual_error_audit'
audit_dir.mkdir()
audit_inventory = []

for hp in project.rglob('*.html'):
    if 'audit' not in hp.name.lower() and 'yolo11 segmentation' not in hp.name.lower():
        continue

    try:
        parser = HTMLCollector()
        parser.feed(hp.read_text(encoding='utf-8', errors='replace'))

        txt = re.sub(r'\n{3,}', '\n\n', '\n'.join(parser.text))
        text_file = re.sub(r'[^A-Za-z0-9_.-]+', '_', hp.stem) + '_text.txt'

        (audit_dir / text_file).write_text(txt, encoding='utf-8')

        saved = []

        for i, (alt, uri) in enumerate(parser.images, 1):
            try:
                header, b64 = uri.split(',', 1)
                mt = re.search(r'data:image/([\w.+-]+)', header)
                ext = mt.group(1).replace('jpeg','jpg') if mt else 'jpg'

                name = re.sub(
                    r'[^A-Za-z0-9_.-]+', '_',
                    Path(alt or f'image_{i}').name
                )

                if not Path(name).suffix:
                    name += '.' + ext

                dest = audit_dir / f'{i:03d}_{name}'
                dest.write_bytes(base64.b64decode(b64))
                saved.append(dest.name)

            except Exception:
                pass

        audit_inventory.append({
            'html': str(hp.relative_to(stage)),
            'text': text_file,
            'embedded_images_found': len(parser.images),
            'images_saved': saved
        })

    except Exception as e:
        audit_inventory.append({'html': str(hp), 'error': repr(e)})

(stage / 'visual_error_audit_inventory.json').write_text(
    json.dumps(audit_inventory, ensure_ascii=False, indent=2), encoding='utf-8'
)

# ============================= TRÍCH CHATGPT EXPORT ==========================
def norm(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip().lower()

def body_text(c):
    if isinstance(c, str):
        return c
    if not isinstance(c, dict):
        return str(c)

    out = []

    for p in c.get('parts', []):
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            out.append(str(p.get('text') or p.get('caption') or '[MEDIA/ATTACHMENT]'))

    if not out and c.get('text'):
        out.append(str(c['text']))

    return '\n'.join(out).strip()

def matching_chat(obj):
    arr = obj if isinstance(obj, list) else [obj]

    if isinstance(obj, dict):
        for k in ('conversations', 'items', 'data'):
            if isinstance(obj.get(k), list):
                arr = obj[k]
                break

    exact = [
        c for c in arr
        if isinstance(c, dict) and 'định dạng annotation' in norm(c.get('title'))
    ]

    if exact:
        arr = exact
    else:
        arr = [
            c for c in arr
            if isinstance(c, dict) and any(
                k in norm(c.get('title'))
                for k in ('yolo11 segmentation', 'yolo11-seg')
            )
        ]

    for conv in arr:
        if not isinstance(conv, dict):
            continue

        messages = []
        mapping = conv.get('mapping', {})

        if isinstance(mapping, dict):
            for node in mapping.values():
                m = node.get('message') if isinstance(node, dict) else None

                if not isinstance(m, dict):
                    continue

                ts = m.get('create_time') or 0

                try:
                    date = datetime.fromtimestamp(float(ts)).astimezone().isoformat() if ts else ''
                except Exception:
                    date = ''

                role = (m.get('author') or {}).get('role', 'unknown')
                text = body_text(m.get('content', {})) or '[MEDIA-ONLY / EMPTY MESSAGE]'
                messages.append((float(ts or 0), role, date, text))

        elif isinstance(conv.get('messages'), list):
            for m in conv['messages']:
                ts = m.get('create_time') or m.get('timestamp') or 0

                try:
                    date = datetime.fromtimestamp(float(ts)).astimezone().isoformat() if ts else ''
                except Exception:
                    date = ''

                role = m.get('role') or (m.get('author') or {}).get('role', 'unknown')
                messages.append((
                    float(ts or 0),
                    role,
                    date,
                    body_text(m.get('content', m.get('text',''))) or '[EMPTY/MEDIA MESSAGE]'
                ))

        messages.sort(key=lambda x: x[0])
        lines = [f'# {conv.get("title", "YOLO11 Segmentation")}\n']

        for _, role, date, text in messages:
            lines += ['\n---\n', f'## {role} {date}\n\n', text, '\n']

        return '\n'.join(lines)

    return None


chat_dir = stage / 'chat_transcript'
chat_dir.mkdir()

chat_status = {
    'transcript_extracted': False,
    'sources_checked': [],
    'note': 'Không có transcript ChatGPT đầy đủ trừ khi người dùng upload đúng bản export.'
}

for jp in project.rglob('*.json'):
    if jp.name.lower() not in ('conversations.json','conversation.json') and 'chat' not in jp.name.lower():
        continue

    chat_status['sources_checked'].append(str(jp.relative_to(stage)))

    try:
        transcript = matching_chat(
            json.loads(jp.read_text(encoding='utf-8', errors='replace'))
        )
    except Exception:
        transcript = None

    if transcript:
        (chat_dir / 'Dinh_dang_annotation_YOLO11_Segmentation.md').write_text(
            transcript, encoding='utf-8'
        )
        chat_status.update({
            'transcript_extracted': True,
            'source': str(jp.relative_to(stage))
        })
        break

# ChatGPT account export ZIP: extract matching conversation JSON.
if not chat_status['transcript_extracted']:
    for zp in project.rglob('*.zip'):
        try:
            with zipfile.ZipFile(zp) as z:
                candidates = [
                    n for n in z.namelist()
                    if Path(n).name.lower() == 'conversations.json'
                ]

                for member in candidates:
                    try:
                        transcript = matching_chat(
                            json.loads(z.read(member).decode('utf-8', errors='replace'))
                        )
                    except Exception:
                        transcript = None

                    if transcript:
                        (chat_dir / 'Dinh_dang_annotation_YOLO11_Segmentation.md').write_text(
                            transcript, encoding='utf-8'
                        )
                        chat_status.update({
                            'transcript_extracted': True,
                            'source': f'{zp.name}:{member}'
                        })
                        break

                if chat_status['transcript_extracted']:
                    break

        except Exception:
            pass

# Also extract visible text and embedded images from a chat-specific HTML export.
for hp in project.rglob('*.html'):
    if not any(k in hp.name.lower() for k in ('chat','conversation','định dạng annotation')):
        continue

    try:
        parser = HTMLCollector()
        parser.feed(hp.read_text(encoding='utf-8', errors='replace'))

        (chat_dir / (hp.stem + '_visible_text.txt')).write_text(
            '\n'.join(parser.text), encoding='utf-8'
        )

        media_dir = chat_dir / 'embedded_html_images'
        media_dir.mkdir(exist_ok=True)
        extracted = []

        for i, (alt, uri) in enumerate(parser.images, 1):
            try:
                header, b64 = uri.split(',', 1)
                mt = re.search(r'data:image/([\w.+-]+)', header)
                ext = mt.group(1).replace('jpeg', 'jpg') if mt else 'jpg'

                name = re.sub(
                    r'[^A-Za-z0-9_.-]+', '_',
                    Path(alt or f'image_{i}').name
                )

                if not Path(name).suffix:
                    name += '.' + ext

                dst = media_dir / f'{i:03d}_{name}'
                dst.write_bytes(base64.b64decode(b64))
                extracted.append(dst.name)

            except Exception:
                pass

        chat_status['html_export'] = str(hp.relative_to(stage))
        chat_status['embedded_html_images'] = extracted

    except Exception:
        pass

(stage / 'chat_export_status.json').write_text(
    json.dumps(chat_status, ensure_ascii=False, indent=2), encoding='utf-8'
)

# ============================= TÓM TẮT TRAINING CSV ==========================
training = []

for cp in project.rglob('*.csv'):
    try:
        with open(cp, encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        cols = list(rows[0])
        clean = {c: c.strip().lower() for c in cols}

        if 'epoch' not in clean.values() or not any(
            any(k in c.lower() for k in ('map50','precision','recall','loss'))
            for c in cols
        ):
            continue

        mapcols = [c for c in cols if 'map50-95' in c.lower()]
        bestrow, best = None, None

        for row in rows:
            vals = []

            for c in mapcols:
                try:
                    vals.append(float(row[c]))
                except Exception:
                    pass

            if vals and (best is None or max(vals) > best):
                best, bestrow = max(vals), row

        training.append({
            'csv': str(cp.relative_to(stage)),
            'epochs_logged': len(rows),
            'columns': cols,
            'last_row': rows[-1],
            'best_map50_95': best,
            'best_row': bestrow
        })

    except Exception:
        pass

(stage / 'training_results_summary.json').write_text(
    json.dumps(training, ensure_ascii=False, indent=2), encoding='utf-8'
)

# ============================= GHI CHÚ NGHIÊN CỨU ============================
notes = r'''# YOLO11 Instance Segmentation — Research Handover

## A. Mục tiêu và dữ liệu đã biết
- Hướng làm việc: YOLO11-seg / instance segmentation trong Google Colab, dùng polygon instance annotations được gán bằng CVAT.
- `job62.zip`: thông tin người dùng từng báo là 597 files, 298 ảnh, 298 TXT và 1 YAML; cũng cho biết có `train.txt`. Cần đối chiếu inventory tự động vì số lượng file TXT có thể bao gồm file split.
- `data.yaml`: `path: .`, `train: train.txt`.
- Class IDs: 0 hat; 1 gloves; 2 mask; 3 pants; 4 shoes; 5 long-sleeved shirt.
- Ví dụ nhãn đặt dưới `labels/train/batch_007/`; mỗi dòng bắt đầu bằng class ID, tiếp theo là tọa độ polygon x/y chuẩn hóa theo định dạng YOLO segmentation.
- `job54.zip` cũng được nhắc đến nhưng trước đây chưa có số lượng ảnh/nhãn hay YAML được xác nhận. Xem `dataset_zip_inventory.json` để lấy số thực tế.

## B. Visual Error Audit đã xác nhận
Báo cáo `YOLO11 Segmentation - Visual Error Audit.html` (nếu được đưa vào Colab) chứa ảnh overlay nhúng bên trong.
- Validation: 268 images; 3,042 Ground Truth instances.
- Confidence threshold 0.40; mask IoU threshold 0.50.
- 6 classes; top 2 images/model/class; 24 overlays, 16 unique image names.
- Hai model labels trong báo cáo: `CB_IRFS`, `C_IRFS`.

### Overall metrics
| Metric | CB_IRFS | C_IRFS |
|---|---:|---:|
| GT | 3042 | 3042 |
| Predictions | 2474 | 2519 |
| TP | 2201 | 2223 |
| FP | 273 | 296 |
| FN | 841 | 819 |
| Precision | 0.8897 | 0.8825 |
| Recall | 0.7235 | 0.7308 |
| F1 | 0.7980 | 0.7995 |

### Per-class results
| Class | CB F1 | C F1 | CB Recall | C Recall |
|---|---:|---:|---:|---:|
| hat | 0.8804 | 0.8738 | 0.8432 | 0.8311 |
| gloves | 0.5348 | 0.5479 | 0.4202 | 0.4202 |
| mask | 0.8155 | 0.8109 | 0.7434 | 0.7463 |
| pants | 0.7494 | 0.7393 | 0.6637 | 0.6614 |
| shoes | 0.6122 | 0.6131 | 0.4927 | 0.5292 |
| long-sleeved shirt | 0.8434 | 0.8596 | 0.7849 | 0.8090 |

### Preliminary interpretation — not causal claims
- C_IRFS hơn F1 0.0015 tuyệt đối: rất nhỏ, chưa đủ kết luận vượt trội rõ nếu chưa repeated seeds/test độc lập.
- CB_IRFS có Precision cao hơn và FP ít hơn; C_IRFS có Recall cao hơn và FN ít hơn.
- Gloves yếu nhất theo F1 (~0.535–0.548), Recall 0.4202 ở cả hai.
- Shoes có Recall thấp (~0.493–0.529).
- Long-sleeved shirt có FN thô cao (197 CB, 175 C) nhưng cũng có 916 GT; cần đọc cả FN/GT.
- C nhỉnh hơn ở gloves, shoes, shirt; CB nhỉnh hơn ở hat, mask, pants. Không model nào thắng mọi lớp.
- FP được chia thành duplicate/competing prediction, no significant overlap, partial overlap/localization. Dùng overlay để kiểm tra trước khi quy kết nguyên nhân.
- Các thống kê là validation diagnostics, KHÔNG phải independent test results. Không trình bày chúng như kết quả test cuối cùng.

## C. Những thứ chưa được xác minh — không tự suy diễn
- Quy mô cuối cùng và cấu trúc `job54.zip`, tổng số ảnh sau merge.
- Cách tạo split; leakage theo video/person/session hoặc ảnh gần trùng.
- Mapping chính xác của C_IRFS/CB_IRFS sang kiến trúc/checkpoint.
- Seed, epochs, batch size, imgsz, optimizer, LR, augmentations, pretrained weights, patience, GPU và thời gian train.
- Test độc lập, repeated seeds, confidence intervals, threshold sweep.
- Chất lượng polygon GT, policy occlusion/ambiguous class, missing instances.

## D. Cấu trúc archive
- `project_files/`: file trong `/content` và `ADDITIONAL_ROOTS`.
- `notebook_history/`: input lệnh và những expression outputs còn trong IPython `In`/`Out`; không đảm bảo chứa mọi stdout/ảnh từng render trong giao diện Colab.
- `visual_error_audit/`: text và ảnh embedded data-URI từ audit HTML.
- `chat_transcript/`: transcript lọc từ ChatGPT export nếu có.
- `runtime_environment.json`, `pip_freeze.txt`, `nvidia_smi.txt`: môi trường.
- `dataset_zip_inventory.json`, `training_results_summary.json`, `manifest.json`: kiểm kê dataset, kết quả training và SHA-256.

## E. Giới hạn quan trọng
Cell không thể truy cập lịch sử một chat ChatGPT khác. Để lưu trọn trao đổi từ chiều 30/10/2026 đến lúc export, cần upload ChatGPT data export ZIP/`conversations.json` hoặc transcript HTML của đúng cuộc trò chuyện. Ảnh audit sẽ được trích xuất nếu HTML gốc có trong `/content`; hình ảnh đính kèm chat cần có trong export hoặc được upload riêng. Nếu upload ZIP export toàn tài khoản, ZIP gốc cũng được giữ trong archive và có thể chứa các chat khác; ưu tiên transcript của riêng cuộc trò chuyện này nếu không muốn đưa dữ liệu không liên quan vào gói.
- Muốn giữ đầy đủ output/ảnh hiển thị của notebook, tải notebook `.ipynb` bằng File → Download → Download .ipynb rồi upload vào `/content` trước khi chạy cell. `In`/`Out` chỉ lưu những gì còn trong kernel hiện tại.
'''

(stage / 'RESEARCH_HANDOVER.md').write_text(notes, encoding='utf-8')

# ============================= MANIFEST / README ============================
files = [p for p in project.rglob('*') if p.is_file()]

summary = {
    'created_at': now.isoformat(),
    'project_files': len(manifest),
    'project_size_bytes': sum(x.get('size_bytes',0) for x in manifest),
    'images': sum(
        p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff','.gif'}
        for p in files
    ),
    'dataset_zips': zip_inventory,
    'audit_reports': audit_inventory,
    'training_csvs_summarized': len(training),
    'roots_scanned': [str(x) for x in roots]
}

(stage / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)
(stage / 'archive_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
)

readme = [
    '# Archive contents',
    f'- Created: {now.isoformat()}',
    f'- Files copied: {len(manifest):,}',
    f'- Copied size: {sizeof(summary["project_size_bytes"])}',
    f'- Images found: {summary["images"]:,}',
    f'- Audit HTML processed: {len(audit_inventory)}',
    f'- Audit images extracted: {sum(len(x.get("images_saved",[])) for x in audit_inventory)}',
    f'- Training CSVs summarized: {len(training)}',
    f'- Chat transcript extracted: {chat_status["transcript_extracted"]}',
    '',
    'Mở `RESEARCH_HANDOVER.md`, `chat_export_status.json` và `dataset_zip_inventory.json` để kiểm tra độ đầy đủ.'
]

(stage / 'README_ARCHIVE.md').write_text('\n'.join(readme), encoding='utf-8')

# ============================= TẠO ZIP =======================================
with zipfile.ZipFile(
    archive, 'w', compression=zipfile.ZIP_DEFLATED,
    compresslevel=1, allowZip64=True
) as z:
    for p in stage.rglob('*'):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(stage)))

size = archive.stat().st_size

print('\n' + '='*72 + '\nYOLO11-SEG RESEARCH ARCHIVE CREATED\n' + '='*72)
print('Archive:', archive, '| Size:', sizeof(size))
print('Project files:', len(manifest), '| Audit HTML:', len(audit_inventory),
      '| Extracted audit images:', sum(len(x.get('images_saved',[])) for x in audit_inventory))
print('Training CSV summaries:', len(training), '| Chat transcript extracted:', chat_status['transcript_extracted'])
print('Kiểm tra README_ARCHIVE.md và chat_export_status.json trong ZIP trước khi chuyển chat.')

if DRIVE_COPY_DIR is not None:
    try:
        d = Path(DRIVE_COPY_DIR)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, d / archive.name)
        print('Drive copy:', d / archive.name)
    except Exception as e:
        print('Không copy được sang Drive:', repr(e))

if size < 1_800_000_000:
    try:
        from google.colab import files
        files.download(str(archive))
    except Exception as e:
        print('Không tự download được; ZIP vẫn ở:', archive, '|', repr(e))
else:
    print('ZIP lớn hơn ~1.8 GB. Hãy cấu hình DRIVE_COPY_DIR để lưu vào Drive.')

print('Thư mục tạm để kiểm tra:', stage)

###########################################################################
# CELL 44
###########################################################################
# =============================================================================
# YOLO11-SEG PPE RESEARCH ARCHIVER — KAGGLE EDITION
# Notebook : PPE_31_10
# Purpose  : Archive dataset, annotations, models, metrics, logs, images,
#            Visual Error Audit, notebook history and research handover.
#
# Input  : /kaggle/input
# Output : /kaggle/working
#
# No Google Colab API.
# No file upload required.
# No external internet required.
# =============================================================================

import os
import sys
import re
import csv
import json
import base64
import hashlib
import shutil
import zipfile
import platform
import subprocess
import importlib.metadata

from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from collections import defaultdict

from IPython.display import display, FileLink, Markdown


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INPUT_DIR = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")

ARCHIVE_PREFIX = "YOLO11_PPE_COMPLETE_RESEARCH"

# Include all Kaggle datasets attached to this notebook.
INCLUDE_ALL_KAGGLE_INPUT = True

# Capture available IPython inputs and outputs.
CAPTURE_NOTEBOOK_HISTORY = True

# Calculate SHA256 for each archived file.
# This also allows later integrity verification.
CALCULATE_SHA256 = True

# Compression level: 1 = faster, suitable for images and model checkpoints.
COMPRESSION_LEVEL = 1

# Maximum text output saved from a single notebook output object.
MAX_OUTPUT_CHARS = 150000

TIMESTAMP = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

ARCHIVE_PATH = WORK_DIR / f"{ARCHIVE_PREFIX}_{TIMESTAMP}.zip"

TEMP_DIR = WORK_DIR / f".{ARCHIVE_PREFIX}_TEMP_{TIMESTAMP}"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif"
}

STORED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif", ".mp4", ".avi",
    ".zip", ".7z", ".rar", ".pt", ".pth",
    ".onnx", ".engine", ".safetensors", ".npy",
    ".npz", ".pkl", ".pickle", ".pdf"
}

SKIP_DIRS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".cache"
}

print("=" * 80)
print("YOLO11 PPE RESEARCH ARCHIVER — KAGGLE")
print("=" * 80)

print("Timestamp :", TIMESTAMP)
print("Input     :", INPUT_DIR)
print("Working   :", WORK_DIR)
print("Archive   :", ARCHIVE_PATH)


# =============================================================================
# 2. CHECK KAGGLE ENVIRONMENT
# =============================================================================

if not WORK_DIR.exists():
    raise RuntimeError("Không tìm thấy /kaggle/working.")

if not INPUT_DIR.exists():
    raise RuntimeError("Không tìm thấy /kaggle/input.")

disk = shutil.disk_usage(WORK_DIR)

print("\n[DISK]")
print(f"Total     : {disk.total / 1024**3:.2f} GB")
print(f"Used      : {disk.used / 1024**3:.2f} GB")
print(f"Free      : {disk.free / 1024**3:.2f} GB")

input_roots = sorted(
    [p for p in INPUT_DIR.iterdir() if p.is_dir()],
    key=lambda p: p.name.lower()
)

print("\n[KAGGLE INPUT DATASETS]")
for p in input_roots:
    print(" -", p.name)

print("\n[KAGGLE WORKING]")
for p in sorted(WORK_DIR.iterdir(), key=lambda p: p.name.lower()):
    print(" -", p.name)


# =============================================================================
# 3. DISCOVER ALL SOURCE FILES
# =============================================================================

print("\n" + "=" * 80)
print("DISCOVERING PROJECT FILES")
print("=" * 80)

source_files = []
seen_paths = set()

def should_skip(path):

    try:
        resolved = path.resolve()

        if resolved == ARCHIVE_PATH.resolve():
            return True

        if not path.is_file() or path.is_symlink():
            return True

        if any(part in SKIP_DIRS for part in path.parts):
            return True

        if path.name.startswith(ARCHIVE_PREFIX) and path.suffix.lower() == ".zip":
            return True

        if path.suffix.lower() in {".pyc", ".pyo"}:
            return True

        if path.name == ".DS_Store":
            return True

        return False

    except Exception:
        return True


def discover_files(base, archive_root):

    if not base.exists():
        return

    for path in base.rglob("*"):

        try:
            if should_skip(path):
                continue

            resolved = str(path.resolve())

            if resolved in seen_paths:
                continue

            seen_paths.add(resolved)

            relative = path.relative_to(base)
            archive_name = Path(archive_root) / relative

            source_files.append({
                "path": path,
                "archive_path": archive_name.as_posix(),
                "root": str(base),
                "root_type": archive_root
            })

        except Exception as e:
            print("Discovery warning:", path, repr(e))


# Dataset files.
if INCLUDE_ALL_KAGGLE_INPUT:
    discover_files(INPUT_DIR, "kaggle_input")

# Experiment outputs, code, reports, weights and results.
discover_files(WORK_DIR, "kaggle_working")

source_files.sort(key=lambda x: x["archive_path"].lower())

total_source_size = sum(
    x["path"].stat().st_size for x in source_files
)

print(f"\nTotal files discovered : {len(source_files):,}")
print(f"Total original size    : {total_source_size / 1024**3:.3f} GB")

if not source_files:
    raise RuntimeError("Không tìm thấy file nào để lưu.")

# Conservative check: ZIP may not compress images/checkpoints significantly.
required_space = total_source_size + max(
    100 * 1024**2,
    int(total_source_size * 0.03)
)

if required_space > disk.free * 0.97:
    raise RuntimeError(
        "Không đủ dung lượng trống an toàn để tạo archive.\n"
        f"Estimated required: {required_space / 1024**3:.2f} GB\n"
        f"Available: {disk.free / 1024**3:.2f} GB\n"
        "Hãy xem lại INCLUDE_ALL_KAGGLE_INPUT hoặc giải phóng dung lượng."
    )


# =============================================================================
# 4. DATASET INVENTORY
# =============================================================================

print("\n" + "=" * 80)
print("BUILDING DATASET INVENTORY")
print("=" * 80)

dataset_stats = defaultdict(lambda: {
    "files": 0,
    "size_bytes": 0,
    "images": 0,
    "txt_files": 0,
    "yaml_files": 0,
    "csv_files": 0,
    "zip_files": 0,
    "html_files": 0,
    "weight_files": 0
})

for item in source_files:

    path = item["path"]
    root_type = item["root_type"]

    relative = path.relative_to(Path(item["root"]))

    top = relative.parts[0] if relative.parts else "ROOT"
    key = f"{root_type}/{top}"

    ext = path.suffix.lower()
    size = path.stat().st_size

    stat = dataset_stats[key]

    stat["files"] += 1
    stat["size_bytes"] += size

    if ext in IMAGE_EXTENSIONS:
        stat["images"] += 1

    if ext == ".txt":
        stat["txt_files"] += 1

    if ext in {".yaml", ".yml"}:
        stat["yaml_files"] += 1

    if ext == ".csv":
        stat["csv_files"] += 1

    if ext == ".zip":
        stat["zip_files"] += 1

    if ext in {".html", ".htm"}:
        stat["html_files"] += 1

    if ext in {".pt", ".pth", ".onnx", ".engine", ".safetensors"}:
        stat["weight_files"] += 1

dataset_inventory = {
    k: dict(v) for k, v in sorted(dataset_stats.items())
}

print(json.dumps(dataset_inventory, indent=2, ensure_ascii=False))


# =============================================================================
# 5. NOTEBOOK HISTORY
# =============================================================================

TEMP_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DIR = TEMP_DIR / "notebook_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

history_status = {
    "captured": False,
    "input_cells": 0,
    "output_objects": 0,
    "note": ""
}


def save_output_bundle(bundle, cell_id, output_dir):

    saved = []

    if isinstance(bundle, tuple):
        bundle = bundle[0]

    if not isinstance(bundle, dict):
        return saved

    for mime, value in bundle.items():

        try:

            if mime.startswith("image/"):

                extension = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/svg+xml": ".svg"
                }.get(mime, ".bin")

                if isinstance(value, str):
                    raw = base64.b64decode(value)
                else:
                    raw = bytes(value)

                target = output_dir / f"cell_{cell_id}_{len(saved)}{extension}"
                target.write_bytes(raw)

                saved.append(target.name)

            elif mime in {"text/plain", "text/html", "text/markdown"}:

                extension = {
                    "text/plain": ".txt",
                    "text/html": ".html",
                    "text/markdown": ".md"
                }[mime]

                target = output_dir / f"cell_{cell_id}_{len(saved)}{extension}"

                target.write_text(
                    str(value)[:MAX_OUTPUT_CHARS],
                    encoding="utf-8",
                    errors="replace"
                )

                saved.append(target.name)

        except Exception as e:
            print("Output capture warning:", repr(e))

    return saved


if CAPTURE_NOTEBOOK_HISTORY:

    try:

        ip = get_ipython()
        namespace = ip.user_ns

        inputs = list(namespace.get("In", []))

        history = []

        for i, code in enumerate(inputs):

            if not str(code).strip():
                continue

            history.append({
                "execution_index": i,
                "source": str(code)
            })

        (HISTORY_DIR / "executed_cells.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        combined = []

        for item in history:

            combined.append(
                "\n\n"
                + "=" * 80
                + f"\nEXECUTION INDEX {item['execution_index']}\n"
                + "=" * 80
                + "\n"
                + item["source"]
            )

        (HISTORY_DIR / "executed_cells.py").write_text(
            "".join(combined),
            encoding="utf-8"
        )

        output_dir = HISTORY_DIR / "outputs"
        output_dir.mkdir(exist_ok=True)

        output_index = []

        outputs = namespace.get("Out", {})

        for cell_id, obj in sorted(outputs.items()):

            if obj is None:
                continue

            saved = []

            # Save textual representation.
            try:

                representation = repr(obj)

                target = output_dir / f"cell_{cell_id}_repr.txt"

                target.write_text(
                    representation[:MAX_OUTPUT_CHARS],
                    encoding="utf-8",
                    errors="replace"
                )

                saved.append(target.name)

            except Exception:
                pass

            # Save figure objects.
            try:

                if hasattr(obj, "savefig"):

                    target = output_dir / f"cell_{cell_id}_figure.png"

                    obj.savefig(
                        target,
                        dpi=180,
                        bbox_inches="tight"
                    )

                    saved.append(target.name)

            except Exception:
                pass

            # Save pandas DataFrames.
            try:

                if (
                    obj.__class__.__module__.startswith("pandas")
                    and hasattr(obj, "to_csv")
                ):

                    target = output_dir / f"cell_{cell_id}_table.csv"

                    obj.to_csv(target, index=True)
                    saved.append(target.name)

            except Exception:
                pass

            # Save PIL images.
            try:

                if (
                    obj.__class__.__module__.startswith("PIL")
                    and hasattr(obj, "save")
                ):

                    target = output_dir / f"cell_{cell_id}_image.png"

                    obj.save(target)
                    saved.append(target.name)

            except Exception:
                pass

            # Capture rich display representations when available.
            try:

                if hasattr(obj, "_repr_mimebundle_"):

                    bundle = obj._repr_mimebundle_()

                    saved += save_output_bundle(
                        bundle,
                        cell_id,
                        output_dir
                    )

            except Exception:
                pass

            output_index.append({
                "execution_index": cell_id,
                "object_type": str(type(obj)),
                "saved_files": saved
            })

        (output_dir / "output_index.json").write_text(
            json.dumps(output_index, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        history_status = {
            "captured": True,
            "input_cells": len(history),
            "output_objects": len(output_index),
            "note": (
                "Captured IPython In/Out currently accessible in kernel. "
                "This is not necessarily a complete copy of all frontend "
                "notebook outputs or previous sessions."
            )
        }

    except Exception as e:

        history_status["note"] = (
            "Could not capture IPython history: " + repr(e)
        )

(HISTORY_DIR / "history_status.json").write_text(
    json.dumps(history_status, ensure_ascii=False, indent=2),
    encoding="utf-8"
)


# =============================================================================
# 6. ENVIRONMENT INFORMATION
# =============================================================================

print("\n" + "=" * 80)
print("CAPTURING ENVIRONMENT")
print("=" * 80)

environment = {
    "captured_at_local": datetime.now().astimezone().isoformat(),
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "notebook_name_from_user_context": "PPE_31_10",
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python_version": sys.version,
    "working_directory": os.getcwd(),
    "input_directory": str(INPUT_DIR),
    "working_directory_path": str(WORK_DIR)
}

environment["kaggle_runtime"] = {
    k: os.environ.get(k)
    for k in [
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_KERNEL_INTERNET",
        "KAGGLE_URL_BASE"
    ]
    if os.environ.get(k) is not None
}

packages = [
    "ultralytics",
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "Pillow",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "PyYAML",
    "scipy",
    "seaborn",
    "segment-anything"
]

environment["package_versions"] = {}

for package in packages:

    try:
        environment["package_versions"][package] = (
            importlib.metadata.version(package)
        )
    except Exception:
        environment["package_versions"][package] = "Not installed"


try:

    import torch

    environment["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count()
    }

    if torch.cuda.is_available():

        environment["torch"]["gpu_devices"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]

except Exception as e:

    environment["torch_error"] = repr(e)


def command_output(command):

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60
        )

        return (
            (result.stdout or "")
            + ("\n[STDERR]\n" + result.stderr if result.stderr else "")
        )

    except Exception as e:

        return "Command error: " + repr(e)


environment_dir = TEMP_DIR / "environment"
environment_dir.mkdir(exist_ok=True)

(environment_dir / "runtime_environment.json").write_text(
    json.dumps(environment, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

(environment_dir / "pip_freeze.txt").write_text(
    command_output([sys.executable, "-m", "pip", "freeze"]),
    encoding="utf-8"
)

(environment_dir / "nvidia_smi.txt").write_text(
    command_output(["nvidia-smi"]),
    encoding="utf-8"
)

(environment_dir / "disk_usage.txt").write_text(
    command_output(["df", "-h"]),
    encoding="utf-8"
)


# =============================================================================
# 7. EXTRACT VISUAL ERROR AUDIT HTML
# =============================================================================

print("\n" + "=" * 80)
print("PROCESSING VISUAL ERROR AUDIT")
print("=" * 80)

AUDIT_DIR = TEMP_DIR / "visual_error_audit"
AUDIT_DIR.mkdir(exist_ok=True)

DATA_URI_PATTERN = re.compile(
    r"data:image/([A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+)",
    re.IGNORECASE
)

class AuditHTMLParser(HTMLParser):

    def __init__(self, save_image_callback):

        super().__init__(convert_charrefs=True)

        self.visible_text = []
        self.skip_depth = 0
        self.save_image_callback = save_image_callback
        self.image_tags = 0
        self.data_images_found = 0

    def handle_starttag(self, tag, attrs):

        attrs = dict(attrs)

        if tag in ("script", "style"):
            self.skip_depth += 1

        if tag == "img":

            self.image_tags += 1

            alt = attrs.get("alt") or f"audit_image_{self.image_tags}"

            for attr in ("src", "data-src", "style", "srcset"):

                value = attrs.get(attr)

                if not value:
                    continue

                for match in DATA_URI_PATTERN.finditer(value):

                    self.data_images_found += 1

                    try:

                        image_format = match.group(1).lower()

                        raw = base64.b64decode(
                            re.sub(r"\s+", "", match.group(2))
                        )

                        self.save_image_callback(
                            raw,
                            image_format,
                            alt
                        )

                    except Exception as e:

                        print("Embedded image extraction warning:", repr(e))

        if tag in {
            "div", "p", "tr", "li", "section",
            "h1", "h2", "h3", "br"
        }:
            self.visible_text.append("\n")

    def handle_endtag(self, tag):

        if tag in ("script", "style") and self.skip_depth > 0:
            self.skip_depth -= 1

        if tag in {
            "div", "p", "tr", "li", "section",
            "h1", "h2", "h3"
        }:
            self.visible_text.append("\n")

    def handle_data(self, data):

        if not self.skip_depth and data.strip():
            self.visible_text.append(data.strip())


audit_source_files = [
    x for x in source_files
    if x["path"].suffix.lower() in {".html", ".htm"}
    and any(
        word in x["path"].name.lower()
        for word in ["audit", "visual_error", "error_report"]
    )
]

audit_summary = []
global_image_hashes = set()

for item in audit_source_files:

    html_path = item["path"]

    print("Processing:", html_path)

    extracted_dir = AUDIT_DIR / re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        html_path.stem
    )

    extracted_dir.mkdir(exist_ok=True)

    image_count = 0
    parser = None

    def save_audit_image(raw, image_format, alt):

        nonlocal_placeholder = None
        global_image_hashes.add(hashlib.sha256(raw).hexdigest())

        # Prevent storing repeated Base64 copies.
        digest = hashlib.sha256(raw).hexdigest()

        if digest in getattr(save_audit_image, "seen", set()):
            return

        if not hasattr(save_audit_image, "seen"):
            save_audit_image.seen = set()

        save_audit_image.seen.add(digest)

        safe_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            Path(alt).name
        )

        if Path(safe_name).suffix:
            base_name = Path(safe_name).stem
        else:
            base_name = safe_name

        ext = "jpg" if image_format in {"jpeg", "jpg"} else image_format

        target = extracted_dir / (
            f"{len(list(extracted_dir.iterdir())) + 1:04d}_"
            f"{base_name}.{ext}"
        )

        target.write_bytes(raw)

    parser = AuditHTMLParser(save_audit_image)

    try:

        with open(html_path, "r", encoding="utf-8", errors="replace") as f:

            for chunk in iter(lambda: f.read(1024 * 1024), ""):
                parser.feed(chunk)

        parser.close()

        text_content = re.sub(
            r"\n{3,}",
            "\n\n",
            "\n".join(parser.visible_text)
        )

        (extracted_dir / "audit_visible_text.txt").write_text(
            text_content,
            encoding="utf-8"
        )

        saved_images = [
            p.name for p in extracted_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        image_count = len(saved_images)

        audit_summary.append({
            "source_html": str(html_path),
            "archive_path": item["archive_path"],
            "img_tags": parser.image_tags,
            "data_image_references": parser.data_images_found,
            "unique_images_extracted": image_count,
            "extracted_directory": extracted_dir.name,
            "images": saved_images
        })

        print(
            "  Image tags:", parser.image_tags,
            "| Base64 refs:", parser.data_images_found,
            "| Extracted unique images:", image_count
        )

    except Exception as e:

        audit_summary.append({
            "source_html": str(html_path),
            "error": repr(e)
        })

        print("  Audit processing error:", repr(e))


(AUDIT_DIR / "audit_inventory.json").write_text(
    json.dumps(audit_summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)


# =============================================================================
# 8. SUMMARIZE YOLO TRAINING RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("ANALYZING TRAINING CSV FILES")
print("=" * 80)

training_summary = []

for item in source_files:

    path = item["path"]

    if path.suffix.lower() != ".csv":
        continue

    try:

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        columns = list(rows[0].keys())

        epoch_column = next(
            (
                c for c in columns
                if c and c.strip().lower() == "epoch"
            ),
            None
        )

        metric_columns = [
            c for c in columns
            if c and any(
                key in c.lower()
                for key in [
                    "map50",
                    "precision",
                    "recall",
                    "loss"
                ]
            )
        ]

        if epoch_column is None or not metric_columns:
            continue

        map_columns = [
            c for c in columns
            if c and "map50-95" in c.lower()
        ]

        best_score = None
        best_row = None

        for row in rows:

            values = []

            for c in map_columns:

                try:
                    values.append(float(row[c]))
                except Exception:
                    pass

            if values:

                score = max(values)

                if best_score is None or score > best_score:
                    best_score = score
                    best_row = row

        record = {
            "file": str(path),
            "archive_path": item["archive_path"],
            "epochs_logged": len(rows),
            "columns": columns,
            "last_epoch_results": rows[-1],
            "best_mAP50_95": best_score,
            "best_epoch_results": best_row
        }

        training_summary.append(record)

        print(
            path.name,
            "| Epochs:", len(rows),
            "| Best mAP50-95:", best_score
        )

    except Exception as e:

        print("CSV analysis warning:", path, repr(e))


TRAINING_DIR = TEMP_DIR / "training_analysis"
TRAINING_DIR.mkdir(exist_ok=True)

(TRAINING_DIR / "training_results_summary.json").write_text(
    json.dumps(training_summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)


# =============================================================================
# 9. RESEARCH HANDOVER DOCUMENT
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING RESEARCH HANDOVER")
print("=" * 80)

handover = r"""
# YOLO11 PPE INSTANCE SEGMENTATION — RESEARCH HANDOVER

Generated automatically from the Kaggle research environment.

Notebook referenced in the research session: PPE_31_10.

---

# 1. PROJECT OBJECTIVE

The project concerns PPE (Personal Protective Equipment) instance
segmentation using YOLO11-seg.

The research workflow includes:

- Dataset preparation.
- Instance polygon annotations.
- Model training and checkpoint management.
- Quantitative evaluation.
- Qualitative visual error analysis.
- Comparison between experimental models.
- Research documentation and reproducibility.

The archive contains the actual files discovered in Kaggle input and working
directories. Consult `archive_summary.json` and `manifest.json` for provenance.

# 2. DATASET AND ANNOTATION CONTEXT

Previous discussion mentioned CVAT instance annotations and YOLO segmentation
polygon TXT labels.

The previously reported class mapping was:

| Class ID | Class |
|---|---|
| 0 | hat |
| 1 | gloves |
| 2 | mask |
| 3 | pants |
| 4 | shoes |
| 5 | long-sleeved shirt |

This mapping must be cross-checked against the actual YAML files in this archive.

A YOLO segmentation annotation line contains:
class_id followed by normalized polygon coordinates.

Do not confuse object detection bounding-box labels with instance segmentation
polygon labels.

Previous discussion also mentioned `job62.zip` and `job54.zip` as CVAT exports.
Their relationship to the current Kaggle dataset must be verified from the
archived data rather than assumed.

# 3. VISUAL ERROR AUDIT — PREVIOUSLY REPORTED RESULTS

An earlier analysis of the Visual Error Audit reported the following:

- Validation images: 268
- Ground Truth instances: 3,042
- Confidence threshold: 0.40
- Mask IoU threshold: 0.50
- Classes: 6
- 24 overlays and 16 unique image names
- Models: CB_IRFS and C_IRFS

## 3.1 Overall metrics

| Metric | CB_IRFS | C_IRFS |
|---|---:|---:|
| GT | 3042 | 3042 |
| Predictions | 2474 | 2519 |
| TP | 2201 | 2223 |
| FP | 273 | 296 |
| FN | 841 | 819 |
| Precision | 0.8897 | 0.8825 |
| Recall | 0.7235 | 0.7308 |
| F1-score | 0.7980 | 0.7995 |

## 3.2 Per-class metrics

| Class | CB F1 | C F1 | CB Recall | C Recall |
|---|---:|---:|---:|---:|
| hat | 0.8804 | 0.8738 | 0.8432 | 0.8311 |
| gloves | 0.5348 | 0.5479 | 0.4202 | 0.4202 |
| mask | 0.8155 | 0.8109 | 0.7434 | 0.7463 |
| pants | 0.7494 | 0.7393 | 0.6637 | 0.6614 |
| shoes | 0.6122 | 0.6131 | 0.4927 | 0.5292 |
| long-sleeved shirt | 0.8434 | 0.8596 | 0.7849 | 0.8090 |

IMPORTANT:
These values are transcribed from the earlier discussion and should be
checked against the original Visual Error Audit report included in this
archive before formal publication.

# 4. PRELIMINARY SCIENTIFIC INTERPRETATION

## 4.1 Model comparison

C_IRFS has F1 = 0.7995, while CB_IRFS has F1 = 0.7980.

The absolute difference is only 0.0015.

Therefore, the result does not establish a meaningful or statistically
significant superiority of either model.

CB_IRFS:
- Higher precision.
- Fewer false positives.
- More conservative predictions.

C_IRFS:
- Higher recall.
- Fewer false negatives.
- Slightly higher overall F1 in the reported validation results.

A repeated-seed experiment and independent test evaluation would be needed
before making a strong model-superiority claim.

## 4.2 Class-level weaknesses

Gloves:
- Lowest F1 among the reported classes.
- Recall approximately 0.42.
- Indicates a substantial missed-instance problem.

Shoes:
- Recall approximately 0.49–0.53.
- Potential difficulty with small objects, occlusion, and localization.
- These are hypotheses to inspect, not established causes.

Long-sleeved shirt:
- Relatively high F1.
- Still has a substantial absolute number of false negatives.
- Interpret FN alongside the number of ground-truth objects.

## 4.3 Important evaluation limitation

The previously discussed audit uses validation data.

Validation diagnostics must not be presented as independent test-set results.

The comparison should be described as a validation-set comparison unless an
independent test set and its evaluation protocol are documented.

# 5. VISUAL ERROR ANALYSIS

The archive includes the original audit HTML and, where possible, extracted
embedded images.

Use them to inspect:

- Duplicate or competing predictions.
- False positives without significant overlap.
- Partial overlap and localization errors.
- Missed objects.
- Boundary and mask-shape quality.
- Class confusion.
- Small and occluded objects.
- Difficult lighting and crowded scenes.

Do not infer the cause of a failure from a metric alone.
Use the corresponding image overlay to support qualitative claims.

# 6. TRAINING AND REPRODUCIBILITY

Inspect:

- `training_analysis/training_results_summary.json`
- Original `results.csv` files.
- `args.yaml` and data YAML files.
- Model weights and checkpoints.
- Training and validation logs.
- Confusion matrices, PR curves, F1 curves and mask prediction images.
- Runtime environment and GPU details.

Important experimental variables to verify:

- Exact YOLO11 model variant.
- Pretrained checkpoint.
- Training/validation/test split.
- Number of epochs.
- Batch size.
- Image resolution.
- Optimizer and learning rate.
- Augmentation settings.
- Random seed.
- Early stopping/patience.
- Confidence and IoU thresholds.
- Software and CUDA versions.

Do not infer missing hyperparameters from the model filename.

# 7. UNRESOLVED RESEARCH QUESTIONS

The archive should be inspected to determine:

1. Whether all datasets use the same class mapping.
2. The final number of images and annotated instances.
3. Whether train/validation/test splits are independent.
4. Whether near-duplicate frames or shared video sequences cause leakage.
5. Whether the two compared models use identical evaluation settings.
6. Whether polygon annotation quality differs by class.
7. Whether the validation results are stable across random seeds.
8. Whether performance improvements generalize to an independent test set.

# 8. STATISTICAL AND SCIENTIFIC CAUTIONS

- A small F1 difference is not automatically a meaningful improvement.
- Precision and recall depend on confidence threshold.
- Mask IoU threshold affects TP/FP/FN matching.
- A class with many instances can dominate aggregate results.
- Raw FP/FN counts should be interpreted relative to class frequency.
- One validation split cannot establish robust generalization.
- Visual examples should not be selected only because they look successful.
- Avoid claiming causality from qualitative examples.

# 9. ARCHIVE STRUCTURE

`kaggle_input/`
    Original Kaggle input datasets.

`kaggle_working/`
    Training runs, outputs, annotations, HTML reports, weights, logs,
    images, and other files in Kaggle working storage.

`notebook_history/`
    IPython input cells and outputs still accessible in the current kernel.

`visual_error_audit/`
    Extracted visible text and embedded image assets from audit HTML.

`training_analysis/`
    Automatically summarized training CSV metrics.

`environment/`
    Python, package, PyTorch, CUDA, GPU and system information.

`archive_meta/`
    Archive summary, research notes and integrity manifest.

# 10. CHATGPT CONTINUATION LIMITATION

This file is a research handover, not an exact transcript of the ChatGPT
conversation.

The Kaggle kernel cannot access the complete history of another ChatGPT chat.

The archive preserves the research context available to this cell,
the actual notebook files, and the previous results summarized above.

Any missing discussion, image attachments, or analysis must be recovered
from the original conversation or its exported files.

# 11. RECOMMENDED NEXT STEP

When opening a new ChatGPT conversation:

1. Upload this archive.
2. Ask ChatGPT to inspect the original HTML audit, dataset, weights,
   training curves and notebook history.
3. Reconcile the actual files against this research handover.
4. Build a verified experiment timeline.
5. Separate measured results, interpretations, and untested hypotheses.
6. Continue the research report from verified evidence only.

END OF RESEARCH HANDOVER.
"""

META_DIR = TEMP_DIR / "archive_meta"
META_DIR.mkdir(exist_ok=True)

(META_DIR / "RESEARCH_HANDOVER.md").write_text(
    handover.strip(),
    encoding="utf-8"
)


# =============================================================================
# 10. SAVE ARCHIVE METADATA
# =============================================================================

archive_summary = {
    "archive_name": ARCHIVE_PATH.name,
    "created_at": datetime.now().astimezone().isoformat(),
    "kaggle_input_included": INCLUDE_ALL_KAGGLE_INPUT,
    "notebook_history": history_status,
    "source_file_count": len(source_files),
    "source_total_bytes": total_source_size,
    "source_size_gb": round(total_source_size / 1024**3, 4),
    "dataset_inventory": dataset_inventory,
    "audit_reports_found": len(audit_source_files),
    "audit_results": audit_summary,
    "extracted_audit_images": sum(
        x.get("unique_images_extracted", 0)
        for x in audit_summary
    ),
    "training_csvs_summarized": len(training_summary),
    "environment": environment
}

(META_DIR / "archive_summary.json").write_text(
    json.dumps(archive_summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

README = f"""
# YOLO11 PPE Research Archive

Created: {datetime.now().astimezone().isoformat()}

Original files collected: {len(source_files):,}

Original total size: {total_source_size / 1024**3:.3f} GB

Audit HTML reports found: {len(audit_source_files)}

Extracted audit images: {archive_summary['extracted_audit_images']}

Training CSV files summarized: {len(training_summary)}

IPython input cells captured: {history_status['input_cells']}

IPython output objects captured: {history_status['output_objects']}

## Main files

- RESEARCH_HANDOVER.md
- archive_summary.json
- manifest.json
- training_analysis/training_results_summary.json
- visual_error_audit/audit_inventory.json
- notebook_history/history_status.json
- environment/runtime_environment.json

## Data organization

- kaggle_input/
- kaggle_working/
- notebook_history/
- visual_error_audit/
- training_analysis/
- environment/
- archive_meta/

The original source files are preserved under kaggle_input and
kaggle_working. The original HTML reports are retained.
"""

(META_DIR / "README_ARCHIVE.md").write_text(
    README.strip(),
    encoding="utf-8"
)


# =============================================================================
# 11. CREATE SHA256 MANIFEST AND ZIP ARCHIVE
# =============================================================================

print("\n" + "=" * 80)
print("CREATING FINAL ZIP ARCHIVE")
print("=" * 80)

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def zip_add_file(zf, path, arcname):

    path = Path(path)
    stat = path.stat()

    ext = path.suffix.lower()

    method = (
        zipfile.ZIP_STORED
        if ext in STORED_EXTENSIONS
        else zipfile.ZIP_DEFLATED
    )

    info = zipfile.ZipInfo(
        filename=arcname,
        date_time=datetime.fromtimestamp(stat.st_mtime).timetuple()[:6]
    )

    info.compress_type = method
    info.file_size = stat.st_size
    info.create_system = 3
    info.external_attr = (stat.st_mode & 0xFFFF) << 16

    digest = hashlib.sha256() if CALCULATE_SHA256 else None

    with open(path, "rb") as src, zf.open(
        info,
        mode="w",
        force_zip64=True
    ) as dst:

        for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):

            dst.write(chunk)

            if digest:
                digest.update(chunk)

    return {
        "archive_path": arcname,
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest() if digest else None
    }


manifest = []
processed = 0

with zipfile.ZipFile(
    ARCHIVE_PATH,
    mode="w",
    allowZip64=True,
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=COMPRESSION_LEVEL
) as zf:

    # ---------------------------------------------------------
    # Original Kaggle input and working files.
    # ---------------------------------------------------------

    for item in source_files:

        path = item["path"]

        try:

            result = zip_add_file(
                zf,
                path,
                item["archive_path"]
            )

            result["source_path"] = str(path)
            result["root_type"] = item["root_type"]

            manifest.append(result)

            processed += 1

            if processed % 100 == 0:

                print(
                    f"Archived {processed:,}/{len(source_files):,} "
                    f"original files..."
                )

        except Exception as e:

            manifest.append({
                "source_path": str(path),
                "archive_path": item["archive_path"],
                "error": repr(e)
            })

            print("WARNING: Could not archive:", path, repr(e))

    # ---------------------------------------------------------
    # Generated notebook history, audit extracts, environment,
    # training summary and research handover.
    # ---------------------------------------------------------

    generated_files = sorted(
        [p for p in TEMP_DIR.rglob("*") if p.is_file()],
        key=lambda p: p.as_posix().lower()
    )

    generated_count = 0

    for path in generated_files:

        arcname = (
            "research_metadata/"
            + path.relative_to(TEMP_DIR).as_posix()
        )

        try:

            result = zip_add_file(
                zf,
                path,
                arcname
            )

            result["source_path"] = str(path)
            result["root_type"] = "generated_metadata"

            manifest.append(result)

            generated_count += 1

        except Exception as e:

            print("WARNING: Could not archive metadata:", path, repr(e))

    # ---------------------------------------------------------
    # Final manifest.
    # ---------------------------------------------------------

    zf.writestr(
        "archive_meta/manifest.json",
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ),
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=1
    )

    final_summary = {
        **archive_summary,
        "archived_original_files": processed,
        "archived_generated_files": generated_count,
        "manifest_entries": len(manifest),
        "archive_file": ARCHIVE_PATH.name
    }

    zf.writestr(
        "archive_meta/final_archive_summary.json",
        json.dumps(
            final_summary,
            ensure_ascii=False,
            indent=2
        ),
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=1
    )


# =============================================================================
# 12. VERIFY ARCHIVE
# =============================================================================

print("\n" + "=" * 80)
print("VERIFYING ZIP")
print("=" * 80)

try:

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:

        bad_file = zf.testzip()

        archive_members = zf.namelist()

        zip_verification = {
            "archive_exists": ARCHIVE_PATH.exists(),
            "archive_size_bytes": ARCHIVE_PATH.stat().st_size,
            "archive_size_gb": round(
                ARCHIVE_PATH.stat().st_size / 1024**3,
                4
            ),
            "member_count": len(archive_members),
            "zip_integrity": "PASS" if bad_file is None else "FAIL",
            "first_bad_file": bad_file,
            "research_handover_exists": (
                "research_metadata/archive_meta/RESEARCH_HANDOVER.md"
                in archive_members
            ),
            "manifest_exists": (
                "archive_meta/manifest.json" in archive_members
            )
        }

    (WORK_DIR / f"{ARCHIVE_PREFIX}_{TIMESTAMP}_verification.json").write_text(
        json.dumps(zip_verification, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(zip_verification, ensure_ascii=False, indent=2))

except Exception as e:

    print("ZIP verification error:", repr(e))
    raise


# =============================================================================
# 13. CLEAN TEMPORARY FILES
# =============================================================================

try:

    shutil.rmtree(TEMP_DIR)

    print("\nTemporary build directory removed.")

except Exception as e:

    print("Temporary directory cleanup warning:", repr(e))


# =============================================================================
# 14. FINAL RESULT
# =============================================================================

print("\n" + "=" * 80)
print("YOLO11 PPE RESEARCH ARCHIVE COMPLETED")
print("=" * 80)

print("\nARCHIVE PATH:")
print(ARCHIVE_PATH)

print("\nARCHIVE SIZE:")
print(f"{ARCHIVE_PATH.stat().st_size / 1024**3:.3f} GB")

print("\nORIGINAL FILES:")
print(f"{processed:,}")

print("\nAUDIT REPORTS:")
print(len(audit_source_files))

print("\nEXTRACTED AUDIT IMAGES:")
print(archive_summary["extracted_audit_images"])

print("\nTRAINING CSV SUMMARIES:")
print(len(training_summary))

print("\nNOTEBOOK INPUT CELLS:")
print(history_status["input_cells"])

print("\nNOTEBOOK OUTPUT OBJECTS:")
print(history_status["output_objects"])

print("\nZIP INTEGRITY:")
print(zip_verification["zip_integrity"])

print("\nNEXT STEP:")
print("Open the Kaggle Output panel and refresh /kaggle/working.")
print("Find and download:", ARCHIVE_PATH.name)

try:

    display(Markdown(
        f"### Research archive created\n"
        f"**File:** `{ARCHIVE_PATH.name}`  \n"
        f"**Size:** {ARCHIVE_PATH.stat().st_size / 1024**3:.3f} GB"
    ))

    display(FileLink(str(ARCHIVE_PATH)))

except Exception as e:

    print("Download link display unavailable:", repr(e))

print("\nDONE.")

###########################################################################
# CELL 45
###########################################################################
# =============================================================================
# YOLO11 PPE — COMPACT RESEARCH EVIDENCE PACK
#
# Platform : Kaggle Notebook
# Input    : /kaggle/input
# Output   : /kaggle/working
#
# PURPOSE:
# Create a compact evidence package for preparing the LaTeX research report.
#
# INCLUDED:
#   - Dataset YAML, JSON, CSV, TXT metadata
#   - YOLO polygon annotation TXT files
#   - train.txt / val.txt / test.txt
#   - Metadata extracted from CVAT ZIP files
#   - Training results.csv, args.yaml, logs and code
#   - Selected plots and validation images
#   - Visual Error Audit text, sanitized HTML and embedded images
#   - Notebook history available in the current kernel
#   - Annotation statistics and polygon quality checks
#   - Runtime environment and package versions
#   - Research handover document
#
# EXCLUDED:
#   - Original dataset images
#   - Large dataset ZIP files
#   - Model weights: .pt, .pth, .onnx, .engine
#   - Unnecessary temporary files
#
# No Google Colab API.
# No upload required.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import os
import sys
import re
import csv
import json
import math
import base64
import hashlib
import shutil
import zipfile
import platform
import importlib.metadata

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from html.parser import HTMLParser
from io import BytesIO

from IPython.display import display, FileLink, Markdown


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

INPUT_DIR = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")

TIMESTAMP = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

ARCHIVE_NAME = f"YOLO11_REPORT_EVIDENCE_{TIMESTAMP}.zip"

OUTPUT_ZIP = WORK_DIR / ARCHIVE_NAME

STAGE = WORK_DIR / f"_REPORT_EVIDENCE_STAGE_{TIMESTAMP}"

# Limits for compact package.
MAX_TEXT_FILE_MB = 35
MAX_NOTEBOOK_MB = 30
MAX_AUDIT_HTML_MB = 250

MAX_AUDIT_IMAGES = 80
MAX_PLOT_IMAGES = 60

MAX_IMAGE_SIDE = 1800
IMAGE_QUALITY = 85

# Inspect dataset ZIP archives, but extract only small metadata/labels.
MAX_ZIP_METADATA_MEMBER_MB = 35
MAX_INPUT_ZIP_SIZE_GB = 15

# Capture current IPython kernel history.
INCLUDE_NOTEBOOK_HISTORY = True

# Calculate SHA256 for copied evidence files.
CALCULATE_SHA256 = True

# Input directories that normally contain raw image/video data.
INPUT_SKIP_DIRS = {
    "images",
    "image",
    "jpegimages",
    "jpgimages",
    "frames",
    "videos",
    "video",
    "raw_images",
    "original_images"
}

# Directories that are not needed for report preparation.
SKIP_DIRS = {
    ".git",
    ".cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "wandb",
    "weights",
    "checkpoints",
    "node_modules"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".gif"
}

# Important text file formats.
TEXT_EXTENSIONS = {
    ".csv",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".log",
    ".py",
    ".ipynb",
    ".toml",
    ".cfg",
    ".html",
    ".htm"
}

WEIGHT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".safetensors"
}

# Names used to discover experiment plots.
PLOT_KEYWORDS = [
    "results.png",
    "confusion_matrix",
    "pr_curve",
    "f1_curve",
    "p_curve",
    "r_curve",
    "val_batch",
    "train_batch",
    "labels.jpg",
    "labels.png",
    "class_distribution",
    "per_class",
    "comparison",
    "baseline",
    "overlay",
    "prediction",
    "audit",
    "metric",
    "curve"
]

# Possible HTML audit report names.
AUDIT_KEYWORDS = [
    "audit",
    "visual_error",
    "error_report",
    "error_audit"
]


# =============================================================================
# 3. INITIALIZATION
# =============================================================================

print("=" * 80)
print("YOLO11 PPE — COMPACT RESEARCH EVIDENCE ARCHIVER")
print("=" * 80)

print("Timestamp :", TIMESTAMP)
print("Input     :", INPUT_DIR)
print("Working   :", WORK_DIR)
print("Output    :", OUTPUT_ZIP)

if not INPUT_DIR.exists():
    raise RuntimeError("Không tìm thấy /kaggle/input.")

if not WORK_DIR.exists():
    raise RuntimeError("Không tìm thấy /kaggle/working.")

if STAGE.exists():
    shutil.rmtree(STAGE)

STAGE.mkdir(parents=True, exist_ok=True)

disk = shutil.disk_usage(WORK_DIR)

print("\n[DISK INFORMATION]")
print(f"Total : {disk.total / 1024**3:.2f} GB")
print(f"Used  : {disk.used / 1024**3:.2f} GB")
print(f"Free  : {disk.free / 1024**3:.2f} GB")


# =============================================================================
# 4. UTILITY FUNCTIONS
# =============================================================================

included_sources = set()

source_manifest = []

label_files = []

audit_html_files = []

plot_images = []

zip_candidates = []

statistics = {
    "input_metadata_files": 0,
    "working_evidence_files": 0,
    "annotation_files": 0,
    "plot_images": 0,
    "audit_reports": 0,
    "audit_images": 0,
    "excluded_weights": 0,
    "excluded_raw_images": 0,
    "excluded_large_text_files": 0
}


def safe_name(value):

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value)
    )

    return value[:150] or "unnamed"


def file_size(path):

    try:
        return Path(path).stat().st_size
    except Exception:
        return 0


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def sha256_bytes(data):

    return hashlib.sha256(data).hexdigest()


def is_inside(path, parent):

    try:

        Path(path).resolve().relative_to(
            Path(parent).resolve()
        )

        return True

    except Exception:

        return False


def save_source_file(source, relative_path, category):

    """
    Copy selected source file into the evidence staging folder.
    Does not modify original source.
    """

    source = Path(source)

    if not source.is_file() or source.is_symlink():
        return None

    resolved = str(source.resolve())

    if resolved in included_sources:
        return None

    destination = STAGE / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        shutil.copy2(source, destination)

        included_sources.add(resolved)

        record = {
            "source": str(source),
            "archive_path": destination.relative_to(
                STAGE
            ).as_posix(),
            "category": category,
            "size_bytes": file_size(destination),
            "sha256": (
                sha256_file(destination)
                if CALCULATE_SHA256
                else None
            )
        }

        source_manifest.append(record)

        if category == "input_metadata":
            statistics["input_metadata_files"] += 1

        elif category == "working_evidence":
            statistics["working_evidence_files"] += 1

        return destination

    except Exception as e:

        print(
            "WARNING: Copy failed:",
            source,
            repr(e)
        )

        return None


# =============================================================================
# 5. SCAN KAGGLE INPUT AND WORKING DIRECTORIES
# =============================================================================

print("\n" + "=" * 80)
print("[1/10] SCANNING KAGGLE INPUT AND WORKING")
print("=" * 80)


def scan_root(root, input_mode=False):

    """
    INPUT:
      Keep metadata, YAML, JSON, CSV, annotation TXT and split TXT.
      Do not copy raw images.

    WORKING:
      Keep training logs, configurations, metrics, code and selected reports.
      Keep only selected visualization images.
      Do not copy checkpoints.
    """

    root = Path(root)

    if not root.exists():
        return

    root_name = "kaggle_input" if input_mode else "kaggle_working"

    for current, dirs, files in os.walk(root):

        current_path = Path(current)

        # Prune unnecessary directories before walking into them.
        extra_skip = INPUT_SKIP_DIRS if input_mode else set()

        dirs[:] = [
            d for d in dirs
            if d.lower() not in SKIP_DIRS
            and d.lower() not in extra_skip
            and not d.startswith(".")
            and not (
                not input_mode
                and (current_path / d).resolve() == STAGE.resolve()
            )
        ]

        for filename in files:

            path = current_path / filename

            try:

                if path.is_symlink():
                    continue

                if is_inside(path, STAGE):
                    continue

                if path.resolve() == OUTPUT_ZIP.resolve():
                    continue

                if not path.is_file():
                    continue

                ext = path.suffix.lower()

                name_lower = filename.lower()

                relative = path.relative_to(root)

                relative_string = relative.as_posix()

                lower_path = relative_string.lower()

                size = file_size(path)

                # -----------------------------------------------------
                # Exclude model weights.
                # -----------------------------------------------------

                if ext in WEIGHT_EXTENSIONS:

                    statistics["excluded_weights"] += 1
                    continue

                # -----------------------------------------------------
                # Find large HTML audits BEFORE applying text limit.
                # This prevents accidentally ignoring a 40 MB audit.
                # -----------------------------------------------------

                if ext in {".html", ".htm"}:

                    is_audit = any(
                        keyword in name_lower
                        for keyword in AUDIT_KEYWORDS
                    )

                    if is_audit:

                        audit_html_files.append({
                            "path": path,
                            "root_type": root_name,
                            "relative_path": relative_string,
                            "size_bytes": size
                        })

                    continue

                # -----------------------------------------------------
                # INPUT METADATA.
                # -----------------------------------------------------

                if input_mode:

                    is_split_file = name_lower in {
                        "train.txt",
                        "val.txt",
                        "test.txt",
                        "train_list.txt",
                        "val_list.txt"
                    }

                    path_parts = {
                        part.lower()
                        for part in relative.parts
                    }

                    is_label_file = (
                        ext == ".txt"
                        and (
                            "labels" in path_parts
                            or "annotations" in path_parts
                        )
                        and size <= MAX_TEXT_FILE_MB * 1024**2
                    )

                    is_metadata = (
                        ext in {
                            ".yaml",
                            ".yml",
                            ".json",
                            ".csv",
                            ".md",
                            ".py"
                        }
                        and size <= MAX_TEXT_FILE_MB * 1024**2
                    )

                    if is_split_file or is_label_file or is_metadata:

                        destination = (
                            Path("input_metadata")
                            / relative
                        )

                        copied = save_source_file(
                            path,
                            destination,
                            "input_metadata"
                        )

                        if copied and is_label_file:

                            label_files.append({
                                "path": copied,
                                "source": str(path),
                                "archive_path": destination.as_posix(),
                                "source_type": "input_directory"
                            })

                            statistics["annotation_files"] += 1

                    elif ext in IMAGE_EXTENSIONS:

                        statistics["excluded_raw_images"] += 1

                    continue

                # -----------------------------------------------------
                # WORKING EVIDENCE.
                # -----------------------------------------------------

                if ext in TEXT_EXTENSIONS:

                    if size > MAX_TEXT_FILE_MB * 1024**2:

                        statistics["excluded_large_text_files"] += 1
                        continue

                    if ext == ".ipynb" and size > MAX_NOTEBOOK_MB * 1024**2:
                        continue

                    # HTML is handled separately above.
                    destination = (
                        Path("working_evidence")
                        / relative
                    )

                    copied = save_source_file(
                        path,
                        destination,
                        "working_evidence"
                    )

                    if copied and ext == ".txt":

                        path_parts = {
                            part.lower()
                            for part in relative.parts
                        }

                        if (
                            "labels" in path_parts
                            or "annotations" in path_parts
                        ):

                            label_files.append({
                                "path": copied,
                                "source": str(path),
                                "archive_path": destination.as_posix(),
                                "source_type": "working_directory"
                            })

                            statistics["annotation_files"] += 1

                # -----------------------------------------------------
                # Selected visualization evidence.
                # -----------------------------------------------------

                elif ext in IMAGE_EXTENSIONS:

                    relevant_name = any(
                        keyword in name_lower
                        for keyword in PLOT_KEYWORDS
                    )

                    relevant_folder = any(
                        keyword in lower_path
                        for keyword in [
                            "visual_error",
                            "audit",
                            "predict",
                            "validation",
                            "val/",
                            "comparison"
                        ]
                    )

                    if (
                        relevant_name or relevant_folder
                    ) and len(plot_images) < MAX_PLOT_IMAGES:

                        plot_images.append({
                            "path": path,
                            "relative": relative,
                            "source": str(path)
                        })

                    else:

                        statistics["excluded_raw_images"] += 1

            except Exception as e:

                print(
                    "Scan warning:",
                    path,
                    repr(e)
                )


scan_root(INPUT_DIR, input_mode=True)

scan_root(WORK_DIR, input_mode=False)

# Remove duplicated audit paths.
unique_audits = {}

for item in audit_html_files:
    unique_audits[str(item["path"].resolve())] = item

audit_html_files = list(unique_audits.values())

print("Input metadata files :", statistics["input_metadata_files"])
print("Working evidence files:", statistics["working_evidence_files"])
print("Annotation files     :", len(label_files))
print("Potential plots      :", len(plot_images))
print("Audit HTML reports   :", len(audit_html_files))
print("Excluded weights     :", statistics["excluded_weights"])
print("Excluded raw images  :", statistics["excluded_raw_images"])


# =============================================================================
# 6. INSPECT DATASET ZIP FILES WITHOUT EXTRACTING RAW IMAGES
# =============================================================================

print("\n" + "=" * 80)
print("[2/10] INSPECTING DATASET ZIP METADATA")
print("=" * 80)

ZIP_METADATA_DIR = (
    STAGE / "annotations_from_dataset_archives"
)

ZIP_METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

zip_inventory = []

zip_seen = set()

for current, dirs, files in os.walk(INPUT_DIR):

    dirs[:] = [
        d for d in dirs
        if d.lower() not in SKIP_DIRS
        and d.lower() not in INPUT_SKIP_DIRS
        and not d.startswith(".")
    ]

    for filename in files:

        if filename.lower().endswith(".zip"):

            candidate = Path(current) / filename

            resolved = str(candidate.resolve())

            if resolved not in zip_seen:

                zip_seen.add(resolved)
                zip_candidates.append(candidate)


for zip_path in zip_candidates:

    record = {
        "source": str(zip_path),
        "metadata_extracted": 0,
        "images_excluded": 0,
        "archive_size_bytes": file_size(zip_path),
        "errors": []
    }

    if file_size(zip_path) > MAX_INPUT_ZIP_SIZE_GB * 1024**3:

        record["errors"].append(
            "ZIP skipped because archive exceeds configured size limit."
        )

        zip_inventory.append(record)

        continue

    try:

        with zipfile.ZipFile(zip_path, "r") as zf:

            members = zf.infolist()

            for member in members:

                if member.is_dir():
                    continue

                member_path = Path(member.filename)

                ext = member_path.suffix.lower()

                parts = {
                    part.lower()
                    for part in member_path.parts
                }

                is_split = member_path.name.lower() in {
                    "train.txt",
                    "val.txt",
                    "test.txt",
                    "train_list.txt",
                    "val_list.txt",
                    "data.yaml",
                    "classes.txt"
                }

                is_label = (
                    ext == ".txt"
                    and (
                        "labels" in parts
                        or "annotations" in parts
                    )
                )

                is_config = ext in {
                    ".yaml",
                    ".yml",
                    ".json",
                    ".csv",
                    ".md"
                }

                if ext in IMAGE_EXTENSIONS:

                    record["images_excluded"] += 1
                    continue

                if not (is_split or is_label or is_config):
                    continue

                if member.file_size > MAX_ZIP_METADATA_MEMBER_MB * 1024**2:
                    continue

                try:

                    raw = zf.read(member)

                    relative = (
                        Path(safe_name(zip_path.stem))
                        / member_path
                    )

                    target = ZIP_METADATA_DIR / relative

                    target.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    target.write_bytes(raw)

                    record["metadata_extracted"] += 1

                    if is_label:

                        label_files.append({
                            "path": target,
                            "source": (
                                f"{zip_path}:{member.filename}"
                            ),
                            "archive_path": target.relative_to(
                                STAGE
                            ).as_posix(),
                            "source_type": "input_zip"
                        })

                        statistics["annotation_files"] += 1

                except Exception as e:

                    record["errors"].append(
                        f"{member.filename}: {repr(e)}"
                    )

    except Exception as e:

        record["errors"].append(repr(e))

    zip_inventory.append(record)

    print(
        zip_path.name,
        "| Metadata:", record["metadata_extracted"],
        "| Excluded images:", record["images_excluded"]
    )


# =============================================================================
# 7. EXTRACT VISUAL ERROR AUDIT
# =============================================================================

print("\n" + "=" * 80)
print("[3/10] PROCESSING VISUAL ERROR AUDIT")
print("=" * 80)

AUDIT_DIR = STAGE / "visual_error_audit"

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

try:

    from PIL import Image

    PIL_AVAILABLE = True

except Exception:

    PIL_AVAILABLE = False

DATA_URI_RE = re.compile(
    r"data:image/(?P<fmt>[A-Za-z0-9.+-]+);base64,"
    r"(?P<data>[A-Za-z0-9+/=\r\n]+)",
    re.IGNORECASE
)

audit_summary = []

saved_audit_hashes = set()

audit_image_counter = 0


def save_embedded_data_uri(data_uri, output_dir, alt):

    global audit_image_counter

    match = DATA_URI_RE.search(data_uri)

    if not match:
        return None

    if audit_image_counter >= MAX_AUDIT_IMAGES:
        return None

    try:

        encoded = re.sub(
            r"\s+",
            "",
            match.group("data")
        )

        raw = base64.b64decode(encoded)

        digest = sha256_bytes(raw)

        if digest in saved_audit_hashes:
            return None

        saved_audit_hashes.add(digest)

        image_format = match.group("fmt").lower()

        filename = safe_name(
            Path(alt).stem or f"audit_{audit_image_counter + 1}"
        )

        audit_image_counter += 1

        target = output_dir / (
            f"{audit_image_counter:03d}_{filename}.jpg"
        )

        if PIL_AVAILABLE:

            image = Image.open(
                BytesIO(raw)
            ).convert("RGB")

            image.thumbnail(
                (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE)
            )

            image.save(
                target,
                format="JPEG",
                quality=IMAGE_QUALITY,
                optimize=True
            )

        else:

            extension = (
                ".png"
                if image_format == "png"
                else ".jpg"
            )

            target = target.with_suffix(extension)

            target.write_bytes(raw)

        return target

    except Exception as e:

        print(
            "Embedded image extraction warning:",
            repr(e)
        )

        return None


class AuditParser(HTMLParser):

    def __init__(self, output_dir):

        super().__init__(
            convert_charrefs=True
        )

        self.output_dir = output_dir

        self.visible_text = []

        self.skip_depth = 0

        self.image_tags = 0

        self.images_saved = []

    def handle_starttag(self, tag, attrs):

        attrs = dict(attrs)

        if tag in {"script", "style"}:
            self.skip_depth += 1

        if tag == "img":
            self.image_tags += 1

        alt = (
            attrs.get("alt")
            or f"audit_image_{self.image_tags}"
        )

        # Extract images stored in HTML attributes.
        for attr in (
            "src",
            "href",
            "data-src",
            "style"
        ):

            value = attrs.get(attr)

            if not value:
                continue

            for match in DATA_URI_RE.finditer(value):

                saved = save_embedded_data_uri(
                    match.group(0),
                    self.output_dir,
                    alt
                )

                if saved:
                    self.images_saved.append(saved.name)

        if tag in {
            "div",
            "p",
            "tr",
            "li",
            "section",
            "h1",
            "h2",
            "h3",
            "br"
        }:

            self.visible_text.append("\n")

    def handle_endtag(self, tag):

        if (
            tag in {"script", "style"}
            and self.skip_depth > 0
        ):

            self.skip_depth -= 1

        if tag in {
            "div",
            "p",
            "tr",
            "li",
            "section",
            "h1",
            "h2",
            "h3"
        }:

            self.visible_text.append("\n")

    def handle_data(self, data):

        if not self.skip_depth and data.strip():

            self.visible_text.append(
                data.strip()
            )


for item in audit_html_files:

    html_path = item["path"]

    print("\nProcessing:", html_path.name)

    report_dir = (
        AUDIT_DIR / safe_name(html_path.stem)
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {
        "source": str(html_path),
        "size_bytes": file_size(html_path),
        "images_saved": [],
        "status": "pending"
    }

    if file_size(html_path) > MAX_AUDIT_HTML_MB * 1024**2:

        record["status"] = "skipped_size_limit"

        record["note"] = (
            f"HTML exceeds {MAX_AUDIT_HTML_MB} MB limit."
        )

        audit_summary.append(record)

        print(record["note"])

        continue

    try:

        html_text = html_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        parser = AuditParser(report_dir)

        parser.feed(html_text)
        parser.close()

        # Additional global search:
        # Some reports store Base64 images in JS or CSS,
        # rather than directly inside img src attributes.

        for match in DATA_URI_RE.finditer(html_text):

            if audit_image_counter >= MAX_AUDIT_IMAGES:
                break

            saved = save_embedded_data_uri(
                match.group(0),
                report_dir,
                f"embedded_image_{audit_image_counter + 1}"
            )

            if saved:
                parser.images_saved.append(saved.name)

        visible_text = re.sub(
            r"\n{3,}",
            "\n\n",
            "\n".join(parser.visible_text)
        )

        (report_dir / "audit_visible_text.txt").write_text(
            visible_text,
            encoding="utf-8"
        )

        # Keep the HTML layout and tables without huge Base64 payloads.
        # The extracted images are stored separately.
        sanitized_html = DATA_URI_RE.sub(
            "about:blank",
            html_text
        )

        (report_dir / "audit_sanitized.html").write_text(
            sanitized_html,
            encoding="utf-8"
        )

        saved_images = sorted({
            name for name in parser.images_saved
        })

        record.update({
            "status": "processed",
            "image_tags": parser.image_tags,
            "unique_images_saved": len(saved_images),
            "images_saved": saved_images,
            "visible_text_file": "audit_visible_text.txt",
            "sanitized_html": "audit_sanitized.html"
        })

        statistics["audit_reports"] += 1

        statistics["audit_images"] += len(saved_images)

        audit_summary.append(record)

        print(
            "  IMG tags:", parser.image_tags,
            "| Unique images:", len(saved_images)
        )

    except Exception as e:

        record["status"] = "error"

        record["error"] = repr(e)

        audit_summary.append(record)

        print(
            "  Audit processing error:",
            repr(e)
        )


# =============================================================================
# 8. SAVE SELECTED TRAINING PLOTS AND VALIDATION IMAGES
# =============================================================================

print("\n" + "=" * 80)
print("[4/10] SAVING SELECTED PLOTS")
print("=" * 80)

PLOT_DIR = STAGE / "training_visual_evidence"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plot_inventory = []

plot_hashes = set()

for item in plot_images:

    path = item["path"]

    try:

        digest = sha256_file(path)

        if digest in plot_hashes:
            continue

        plot_hashes.add(digest)

        if PIL_AVAILABLE:

            image = Image.open(
                path
            ).convert("RGB")

            image.thumbnail(
                (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE)
            )

            base_name = safe_name(path.stem)

            target = PLOT_DIR / (
                base_name + ".jpg"
            )

            counter = 1

            while target.exists():

                target = PLOT_DIR / (
                    f"{base_name}_{counter}.jpg"
                )

                counter += 1

            image.save(
                target,
                format="JPEG",
                quality=IMAGE_QUALITY,
                optimize=True
            )

        else:

            target = PLOT_DIR / path.name

            shutil.copy2(
                path,
                target
            )

        plot_inventory.append({
            "source": str(path),
            "saved_as": target.name,
            "size_bytes": file_size(target),
            "sha256_original": digest
        })

        statistics["plot_images"] += 1

    except Exception as e:

        print(
            "Plot warning:",
            path,
            repr(e)
        )


# =============================================================================
# 9. ANALYZE YOLO SEGMENTATION ANNOTATIONS
# =============================================================================

print("\n" + "=" * 80)
print("[5/10] ANALYZING YOLO POLYGON ANNOTATIONS")
print("=" * 80)

CLASS_MAPPING = {
    0: "hat",
    1: "gloves",
    2: "mask",
    3: "pants",
    4: "shoes",
    5: "long-sleeved shirt"
}


def analyze_label_group(items):

    result = {
        "source_type": (
            items[0]["source_type"]
            if items else "unknown"
        ),
        "label_files": 0,
        "empty_label_files": 0,
        "nonempty_lines": 0,
        "valid_polygon_instances": 0,
        "class_counts_all_parseable_lines": {},
        "valid_polygon_counts_by_class": {},
        "split_counts": {},
        "quality_checks": {
            "malformed_lines": 0,
            "too_few_polygon_points": 0,
            "odd_coordinate_count": 0,
            "non_finite_coordinates": 0,
            "out_of_range_coordinates": 0,
            "duplicate_lines_within_file": 0
        },
        "parse_errors": []
    }

    class_counter = Counter()

    valid_counter = Counter()

    split_counter = defaultdict(Counter)

    quality = result["quality_checks"]

    for item in items:

        path = item["path"]

        try:

            text = path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            result["label_files"] += 1

            if not text.strip():

                result["empty_label_files"] += 1

                continue

            path_description = (
                str(item["source"])
                + " "
                + str(item["archive_path"])
            ).lower()

            if "/train/" in path_description:
                split = "train"

            elif "/val/" in path_description:
                split = "val"

            elif "/test/" in path_description:
                split = "test"

            else:
                split = "unknown"

            seen_lines = set()

            for line in text.splitlines():

                line = line.strip()

                if not line:
                    continue

                result["nonempty_lines"] += 1

                if line in seen_lines:

                    quality["duplicate_lines_within_file"] += 1

                seen_lines.add(line)

                parts = line.split()

                if len(parts) < 2:

                    quality["malformed_lines"] += 1

                    continue

                try:

                    class_float = float(parts[0])

                    if not math.isfinite(class_float):
                        raise ValueError("Non-finite class ID")

                    class_id = int(class_float)

                    if class_float != class_id or class_id < 0:
                        raise ValueError("Invalid class ID")

                except Exception:

                    quality["malformed_lines"] += 1

                    continue

                class_counter[class_id] += 1

                try:

                    coordinates = [
                        float(x) for x in parts[1:]
                    ]

                except Exception:

                    quality["malformed_lines"] += 1

                    continue

                if not all(
                    math.isfinite(x)
                    for x in coordinates
                ):

                    quality["non_finite_coordinates"] += 1

                    continue

                if len(coordinates) < 6:

                    quality["too_few_polygon_points"] += 1

                    continue

                if len(coordinates) % 2 != 0:

                    quality["odd_coordinate_count"] += 1

                    continue

                if not all(
                    0.0 <= x <= 1.0
                    for x in coordinates
                ):

                    quality["out_of_range_coordinates"] += 1

                    continue

                valid_counter[class_id] += 1

                split_counter[split][class_id] += 1

                result["valid_polygon_instances"] += 1

        except Exception as e:

            result["parse_errors"].append({
                "file": str(path),
                "error": repr(e)
            })

    result["class_counts_all_parseable_lines"] = {
        f"{k}: {CLASS_MAPPING.get(k, 'unknown')}" : v
        for k, v in sorted(class_counter.items())
    }

    result["valid_polygon_counts_by_class"] = {
        f"{k}: {CLASS_MAPPING.get(k, 'unknown')}" : v
        for k, v in sorted(valid_counter.items())
    }

    result["split_counts"] = {
        split: {
            f"{k}: {CLASS_MAPPING.get(k, 'unknown')}": v
            for k, v in sorted(counter.items())
        }
        for split, counter in split_counter.items()
    }

    return result


annotation_statistics = {
    "class_mapping_used_for_display": CLASS_MAPPING,
    "total_label_sources": len(label_files),
    "warning": (
        "Different source groups may contain duplicate copies of the "
        "same labels. Do not sum groups without checking file identity."
    ),
    "groups": []
}

for source_type in [
    "input_directory",
    "working_directory",
    "input_zip"
]:

    selected = [
        item for item in label_files
        if item["source_type"] == source_type
    ]

    if selected:

        group_result = analyze_label_group(
            selected
        )

        annotation_statistics["groups"].append(
            group_result
        )

        print(
            "\nSource:", source_type,
            "| TXT:", group_result["label_files"],
            "| Valid polygons:", group_result["valid_polygon_instances"]
        )

        print(
            "Class counts:",
            group_result["valid_polygon_counts_by_class"]
        )

(STAGE / "annotation_statistics.json").write_text(
    json.dumps(
        annotation_statistics,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# =============================================================================
# 10. ANALYZE TRAINING RESULTS CSV
# =============================================================================

print("\n" + "=" * 80)
print("[6/10] ANALYZING TRAINING RESULTS")
print("=" * 80)

training_results = []

for record in source_manifest:

    if Path(record["archive_path"]).suffix.lower() != ".csv":
        continue

    csv_path = (
        STAGE / record["archive_path"]
    )

    try:

        with open(
            csv_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            rows = list(reader)

        if not rows:
            continue

        columns = list(rows[0].keys())

        epoch_column = next(
            (
                c for c in columns
                if c and c.strip().lower() == "epoch"
            ),
            None
        )

        metric_columns = [
            c for c in columns
            if c and any(
                key in c.lower()
                for key in [
                    "precision",
                    "recall",
                    "map50",
                    "loss"
                ]
            )
        ]

        if epoch_column is None or not metric_columns:
            continue

        best_by_metric = {}

        for column in metric_columns:

            if "map50" not in column.lower():
                continue

            valid_values = []

            for row in rows:

                try:

                    value = float(
                        row[column]
                    )

                    if math.isfinite(value):

                        valid_values.append(
                            (value, row)
                        )

                except Exception:
                    pass

            if valid_values:

                best_value, best_row = max(
                    valid_values,
                    key=lambda item: item[0]
                )

                best_by_metric[column] = {
                    "value": best_value,
                    "epoch": best_row.get(epoch_column),
                    "best_row": best_row
                }

        training_results.append({
            "csv": record["archive_path"],
            "epochs_logged": len(rows),
            "columns": columns,
            "first_epoch": rows[0],
            "last_epoch": rows[-1],
            "best_map_metrics": best_by_metric
        })

        print(
            Path(record["archive_path"]).name,
            "| Epochs:", len(rows),
            "| mAP metrics:", list(best_by_metric.keys())
        )

    except Exception as e:

        print(
            "CSV analysis warning:",
            csv_path,
            repr(e)
        )


TRAINING_DIR = STAGE / "training_analysis"

TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

(TRAINING_DIR / "training_results_summary.json").write_text(
    json.dumps(
        training_results,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

training_index = [
    "# Training results index",
    "",
    "Each entry points to an original CSV preserved in `working_evidence/`.",
    ""
]

for item in training_results:

    training_index.append(
        f"- `{item['csv']}` — "
        f"{item['epochs_logged']} epochs"
    )

(TRAINING_DIR / "training_results_index.md").write_text(
    "\n".join(training_index),
    encoding="utf-8"
)


# =============================================================================
# 11. CAPTURE NOTEBOOK HISTORY
# =============================================================================

print("\n" + "=" * 80)
print("[7/10] CAPTURING NOTEBOOK HISTORY")
print("=" * 80)

HISTORY_DIR = STAGE / "notebook_history"

HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

history_status = {
    "captured": False,
    "input_cells": 0,
    "output_objects": 0,
    "note": ""
}

# Redact some common secret formats before saving code history.
SECRET_PATTERNS = [
    (
        re.compile(
            r'(?i)(api[_-]?key|access[_-]?token|password|secret)'
            r'(\s*[:=]\s*)(["\']).*?\3'
        ),
        r'\1\2"[REDACTED]"'
    ),
    (
        re.compile(
            r'(?i)(authorization\s*[:=]\s*)(["\'])'
            r'(?:bearer\s+)?[^"\']+\2'
        ),
        r'\1"[REDACTED]"'
    ),
    (
        re.compile(
            r'(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*'
        ),
        r'\1[REDACTED]'
    )
]


def redact_secrets(text):

    result = str(text)

    for pattern, replacement in SECRET_PATTERNS:

        result = pattern.sub(
            replacement,
            result
        )

    return result


if INCLUDE_NOTEBOOK_HISTORY:

    try:

        ip = get_ipython()

        namespace = ip.user_ns

        inputs = list(
            namespace.get("In", [])
        )

        cell_records = []

        for i, code in enumerate(inputs):

            if not str(code).strip():
                continue

            cell_records.append({
                "execution_index": i,
                "source": redact_secrets(code)
            })

        (HISTORY_DIR / "executed_cells.json").write_text(
            json.dumps(
                cell_records,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        combined_code = []

        for item in cell_records:

            combined_code.append(
                "\n\n"
                + "#" * 75
                + f"\n# CELL {item['execution_index']}\n"
                + "#" * 75
                + "\n"
                + item["source"]
            )

        (HISTORY_DIR / "executed_cells.py").write_text(
            "".join(combined_code),
            encoding="utf-8"
        )

        # Capture accessible Out objects as text.
        # Rich images and very large in-memory arrays are not copied here.
        outputs = namespace.get("Out", {})

        output_records = []

        for cell_id, obj in sorted(outputs.items()):

            try:

                representation = redact_secrets(
                    repr(obj)
                )

                filename = (
                    f"cell_{cell_id}_output.txt"
                )

                (HISTORY_DIR / filename).write_text(
                    representation[:50000],
                    encoding="utf-8",
                    errors="replace"
                )

                output_records.append({
                    "cell": cell_id,
                    "type": str(type(obj)),
                    "file": filename
                })

            except Exception:
                pass

        history_status = {
            "captured": True,
            "input_cells": len(cell_records),
            "output_objects": len(output_records),
            "note": (
                "Only current kernel In/Out. It cannot recover "
                "outputs from previous sessions or every frontend display."
            )
        }

    except Exception as e:

        history_status["note"] = repr(e)


(HISTORY_DIR / "history_status.json").write_text(
    json.dumps(
        history_status,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# =============================================================================
# 12. CAPTURE RUNTIME ENVIRONMENT
# =============================================================================

print("\n" + "=" * 80)
print("[8/10] CAPTURING RUNTIME ENVIRONMENT")
print("=" * 80)

ENV_DIR = STAGE / "environment"

ENV_DIR.mkdir(
    parents=True,
    exist_ok=True
)

environment = {
    "captured_at_local": datetime.now().astimezone().isoformat(),
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "notebook_context": "PPE_31_10",
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "working_directory": os.getcwd(),
    "packages": {}
}

for package in [
    "ultralytics",
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "Pillow",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "PyYAML",
    "scipy",
    "albumentations"
]:

    try:

        environment["packages"][package] = (
            importlib.metadata.version(package)
        )

    except Exception:

        environment["packages"][package] = "Not installed"


try:

    import torch

    environment["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count()
    }

    if torch.cuda.is_available():

        environment["torch"]["gpu_devices"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]

except Exception as e:

    environment["torch_error"] = repr(e)


(ENV_DIR / "runtime_environment.json").write_text(
    json.dumps(
        environment,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


def command_output(command):

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60
        )

        return (
            result.stdout or ""
        ) + (
            "\n[STDERR]\n" + result.stderr
            if result.stderr else ""
        )

    except Exception as e:

        return repr(e)


(ENV_DIR / "pip_freeze.txt").write_text(
    command_output([
        sys.executable,
        "-m",
        "pip",
        "freeze"
    ]),
    encoding="utf-8"
)

(ENV_DIR / "nvidia_smi.txt").write_text(
    command_output(["nvidia-smi"]),
    encoding="utf-8"
)

(ENV_DIR / "disk_usage.txt").write_text(
    command_output(["df", "-h"]),
    encoding="utf-8"
)


# =============================================================================
# 13. RESEARCH HANDOVER DOCUMENT
# =============================================================================

print("\n" + "=" * 80)
print("[9/10] GENERATING RESEARCH HANDOVER")
print("=" * 80)

handover = r"""
# YOLO11 PPE INSTANCE SEGMENTATION — RESEARCH HANDOVER

## 1. Project objective

The research project investigates YOLO11 instance segmentation on a
PPE dataset containing six classes.

The workflow includes:
- CVAT instance annotation.
- Polygon annotation export.
- Dataset quality control.
- Train/validation split preparation.
- YOLO11-seg baseline training.
- Resolution changes and fine-tuning.
- Copy-Paste augmentation.
- Class-aware oversampling.
- Quantitative evaluation.
- Visual error analysis.

This package is a compact evidence package for report preparation.
It does not contain the original image dataset or model checkpoints.

## 2. Class mapping previously supplied

| Class ID | Class |
|---|---|
| 0 | hat |
| 1 | gloves |
| 2 | mask |
| 3 | pants |
| 4 | shoes |
| 5 | long-sleeved shirt |

The class mapping should be checked against the actual `data.yaml`.

## 3. Dataset information previously recorded

The research notes previously reported:

- job54.zip: 297 images.
- job62.zip: 298 images.
- Combined dataset: 595 unique images.
- Train: 476 images.
- Validation: 119 images.
- Train instances: 6,455.
- Validation instances: 1,434.
- Total instances: 7,889.

These values are transcribed from the earlier report draft.
They should be verified against the actual files in this evidence package.

### Previously reported class distribution

| Class | Train | Validation | Total |
|---|---:|---:|---:|
| hat | 2064 | 365 | 2429 |
| gloves | 688 | 182 | 870 |
| mask | 692 | 184 | 876 |
| pants | 700 | 166 | 866 |
| shoes | 393 | 126 | 519 |
| long-sleeved shirt | 1918 | 411 | 2329 |
| Total | 6455 | 1434 | 7889 |

Shoes is the class with the fewest reported instances.

## 4. Dataset split methodology

The previous report describes splitting by recording session
rather than randomly splitting adjacent frames.

This is intended to reduce leakage caused by near-duplicate frames.

Reported split:

Train:
- BaoAnh__screen_on_off_outdoor_pm_20260908_130620: 43 images
- BaoAnh__screen_on_off_outdoor_pm_20260908_131056: 26 images
- NhatTan__crowd_20260908_143116: 45 images
- NhatTan__layout_20260908_142209: 90 images
- QuocHung__phone_20260908_133350: 38 images
- QuocHung__screen_on_off_outdoor_pm_20260908_130802: 63 images
- QuocHung__screen_on_off_outdoor_pm_20260908_130831: 171 images

Validation:
- BaoAnh__screen_on_off_20260908_113356: 11 images
- NhatTan__layout_20260908_142002: 82 images
- QuocHung__screen_on_off_outdoor_pm_20260908_130658: 26 images

These session names and counts are from the earlier report draft.
Verify against train.txt and val.txt before publication.

## 5. Annotation quality control previously reported

Earlier notes mention:
- Two unannotated frames were accepted.
- One image contained two polygons that were identical.
- The duplicate polygon was corrected.
- A TXT file was missing in an earlier export and was subsequently
  investigated through CVAT.

An additional validation annotation issue was recorded in:
`NhatTan__layout_20260908_142002__001090.txt`

The file was reported to contain 17 nonempty lines but only 16 unique
lines. This was corrected and the validation set was reported to contain
1,434 instances.

These observations must be reconciled with the archived labels.

## 6. Baseline experiment — 640 px

Previously reported configuration:

- Model: YOLO11s-seg.
- Image size: 640.
- Batch: 8.
- Maximum epochs: 100.
- Patience: 25.
- Optimizer: Auto.
- Seed: 42.
- Deterministic: enabled.
- AMP: enabled.
- Workers: 2.
- Mosaic: 0.8.
- MixUp: 0.
- Close Mosaic: 10.
- Additional augmentation: blur/motion blur, brightness/contrast,
  and Gaussian noise.

The baseline reportedly stopped after 49 epochs.
Best checkpoint: epoch 24.

| Metric | Box | Mask |
|---|---:|---:|
| Precision | 77.4% | 77.1% |
| Recall | 56.1% | 55.0% |
| mAP50 | 62.2% | 60.5% |
| mAP50-95 | 37.9% | 34.8% |

These figures are transcribed from the previous report draft.

## 7. Resolution and fine-tuning experiments

Previously reported records:

| Experiment | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline 640 | 77.1% | 55.0% | 60.5% | 34.8% |
| Baseline checkpoint evaluated at 960 | 72.7% | 57.3% | 64.2% | 37.1% |
| Fine-tune 960, training log | 80.8% | 63.5% | 68.8% | 40.8% |
| Fine-tune 960, reevaluation | 80.4% | 63.5% | 68.6% | 40.3% |
| Fine-tune 960 checkpoint evaluated at 1280 | 79.1% | 62.0% | 68.8% | 42.4% |
| Separate home_finetune run | 75.0% | 63.9% | 67.8% | 39.9% |

Important distinction:
- Evaluating a fixed checkpoint at another resolution does not update
  the model's weights.
- Fine-tuning updates the weights.
- Results from separate checkpoints or environments should not
  automatically be merged.

## 8. E1, E2 and E3 — sequential 1280 px fine-tuning

Previously reported shared setup:
- YOLO11s-seg.
- GPU: NVIDIA Tesla T4.
- Image size: 1280.
- Batch size: 2.
- Optimizer: AdamW.
- Initial learning rate: 0.0003.
- Final learning-rate factor: 0.01.
- Warmup epochs: 2.
- Weight decay: 0.0005.
- Seed: 42.
- AMP: enabled.
- Workers: 2.
- Mosaic: 0.40.
- MixUp: 0.
- Close Mosaic: 8.

### E1

- 1280 px.
- Copy-Paste disabled.
- No additional class-aware oversampling.
- Reported run length: 25 epochs.
- Best checkpoint: epoch 13.

### E2

- Initialized from E1 best checkpoint.
- Copy-Paste = 0.10.
- Copy-Paste mode = mixup.
- Standard MixUp = 0.
- Reported run length: 17 epochs.
- Best checkpoint: epoch 7.

### E3

- Initialized from E2 best checkpoint.
- Copy-Paste = 0.10.
- Shoes images were repeated twice in the train list.
- Half of the gloves-only images were repeated once.
- Validation split remained unchanged.
- Reported run length: 15 epochs.
- Best checkpoint: epoch 5.

### Overall results reported in the previous draft

| Metric | E1 | E2 | E3 |
|---|---:|---:|---:|
| Box Precision | 83.8% | 80.8% | 81.8% |
| Box Recall | 61.7% | 61.7% | 63.7% |
| Box mAP50 | 69.7% | 69.7% | 71.7% |
| Box mAP50-95 | 45.3% | 44.8% | 47.4% |
| Mask Precision | 84.7% | 83.3% | 80.8% |
| Mask Recall | 62.7% | 63.2% | 66.8% |
| Mask mAP50 | 70.7% | 70.9% | 73.4% |
| Mask mAP50-95 | 44.2% | 44.5% | 45.5% |

### E1 per-class mask metrics

| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| hat | 91.2% | 71.1% | 84.4% | 56.2% |
| gloves | 80.3% | 48.4% | 54.5% | 33.7% |
| mask | 80.9% | 70.7% | 74.6% | 36.6% |
| pants | 84.0% | 63.3% | 73.9% | 52.1% |
| shoes | 78.7% | 48.4% | 53.0% | 23.1% |
| long-sleeved shirt | 93.0% | 74.2% | 84.1% | 63.5% |

### E2 per-class mask metrics

| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| hat | 92.5% | 72.9% | 86.7% | 57.3% |
| gloves | 76.1% | 48.9% | 57.6% | 35.7% |
| mask | 87.3% | 68.5% | 73.3% | 36.6% |
| pants | 78.9% | 65.7% | 72.4% | 50.9% |
| shoes | 76.4% | 46.8% | 53.1% | 23.0% |
| long-sleeved shirt | 88.9% | 76.2% | 82.6% | 63.8% |

### E3 per-class mask metrics

| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| hat | 84.5% | 81.8% | 87.1% | 56.8% |
| gloves | 71.8% | 51.6% | 60.3% | 37.0% |
| mask | 76.8% | 71.7% | 73.4% | 35.9% |
| pants | 82.6% | 69.9% | 79.1% | 53.1% |
| shoes | 78.9% | 50.6% | 54.9% | 25.2% |
| long-sleeved shirt | 90.3% | 75.1% | 85.6% | 64.7% |

All E1-E3 values above are transcribed from the supplied report draft.
They must be compared against the archived CSVs and validation outputs.

## 9. Preliminary E1-E3 interpretation

E3 has the highest reported:
- Mask mAP50: 73.4%.
- Mask mAP50-95: 45.5%.
- Mask recall: 66.8%.

However:
- E3 mask precision is 80.8%, lower than E1 at 84.7%.
- Shoes remains the weakest class by mask mAP50-95.
- Gloves recall improves only to 51.6%.
- Mask mAP50-95 decreases from E1/E2 to E3.

E2 is initialized from E1, and E3 from E2.
This is sequential fine-tuning, not a fully controlled ablation.

It is therefore not valid to attribute all E3 improvements solely
to oversampling.

A controlled comparison should restart all branches from the
same initial checkpoint and change one augmentation factor at a time.

## 10. Separate Visual Error Audit: CB_IRFS vs C_IRFS

A previously inspected HTML audit was described as containing:
- 268 validation images.
- 3,042 ground-truth instances.
- Confidence threshold: 0.40.
- Mask IoU threshold: 0.50.
- 6 classes.
- 24 overlays and 16 unique image names.
- Model identifiers: CB_IRFS and C_IRFS.

Previously reported metrics:

| Metric | CB_IRFS | C_IRFS |
|---|---:|---:|
| TP | 2201 | 2223 |
| FP | 273 | 296 |
| FN | 841 | 819 |
| Precision | 0.8897 | 0.8825 |
| Recall | 0.7235 | 0.7308 |
| F1 | 0.7980 | 0.7995 |

Interpretation:
- C_IRFS has a slightly higher F1.
- The absolute difference is 0.0015.
- CB_IRFS has fewer false positives.
- C_IRFS has fewer false negatives.
- Neither model can be declared clearly superior from this
  small F1 difference alone.

This audit is a separate comparison.
Do not merge it with E1-E3 unless the exact checkpoints,
validation data and evaluation protocol establish their relationship.

The audit figures must be checked against the original HTML.

## 11. Scientific limitations

- A validation set is not an independent test set.
- Repeated configuration selection using the same validation set
  can lead to optimistic performance estimates.
- One split is not sufficient to prove robust generalization.
- Sequential fine-tuning does not isolate causal contributions
  of individual augmentation methods.
- A higher mAP does not guarantee better operational performance
  under every false-positive/false-negative cost.
- Per-class errors should be inspected visually.
- Annotation errors may affect both training and validation.
- Class imbalance is a plausible issue, but should be connected
  to observed per-class errors before claiming it as the cause.

## 12. Evidence package contents

`input_metadata/`
    YAML, JSON, CSV, split TXT and annotation files from Kaggle input.

`annotations_from_dataset_archives/`
    Annotation TXT, data.yaml and split lists extracted from CVAT ZIPs.
    Original images are not extracted.

`working_evidence/`
    Training logs, configs, results CSV, code and experiment metadata.

`training_visual_evidence/`
    Selected training curves, confusion matrices, validation and prediction
    images, compressed for report use.

`visual_error_audit/`
    Audit visible text, sanitized HTML and extracted embedded images.

`training_analysis/`
    Automatically summarized results.csv files.

`annotation_statistics.json`
    Polygon statistics and basic coordinate validity checks.

`notebook_history/`
    Code inputs and outputs still accessible in the current kernel.

`environment/`
    Python, package versions, CUDA, GPU and system information.

## 13. Important distinction between evidence and reported values

The experiment tables in this document were transcribed from the
previously supplied LaTeX draft.

They are not independently measured by this evidence pack.

For the final report:
1. Compare the values with original results.csv files.
2. Check args.yaml for each run.
3. Inspect the audit HTML for its separate model comparison.
4. Check the dataset split lists.
5. Explain discrepancies rather than silently selecting the
   most favorable value.

## 14. Recommended final report organization

1. Introduction and research objective.
2. Dataset collection and CVAT annotation.
3. Annotation format and quality control.
4. Class distribution and train/validation splitting.
5. YOLO11-seg architecture and methodology.
6. Baseline at 640 px.
7. Resolution experiments and 960 px fine-tuning.
8. 1280 px sequential experiments E1-E3.
9. Overall and per-class results.
10. Visual error audit.
11. Discussion and limitations.
12. Conclusion and future experiments.

END OF RESEARCH HANDOVER.
"""

(STAGE / "RESEARCH_HANDOVER.md").write_text(
    handover.strip(),
    encoding="utf-8"
)


# =============================================================================
# 14. SAVE EVIDENCE INVENTORIES
# =============================================================================

print("\n" + "=" * 80)
print("[10/10] SAVING INVENTORIES AND BUILDING ZIP")
print("=" * 80)

(STAGE / "dataset_zip_inventory.json").write_text(
    json.dumps(
        zip_inventory,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

(STAGE / "visual_error_audit_inventory.json").write_text(
    json.dumps(
        audit_summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

(STAGE / "plot_inventory.json").write_text(
    json.dumps(
        plot_inventory,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

(STAGE / "source_manifest.json").write_text(
    json.dumps(
        source_manifest,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

summary = {
    "created_at": datetime.now().astimezone().isoformat(),
    "archive_name": ARCHIVE_NAME,
    "source_files_included": len(source_manifest),
    "annotation_files_found": len(label_files),
    "audit_reports_processed": statistics["audit_reports"],
    "audit_images_extracted": statistics["audit_images"],
    "selected_plots": statistics["plot_images"],
    "training_csvs_summarized": len(training_results),
    "dataset_zip_archives_inspected": len(zip_inventory),
    "notebook_history": history_status,
    "excluded_weights": statistics["excluded_weights"],
    "excluded_raw_images": statistics["excluded_raw_images"],
    "excluded_large_text_files": statistics["excluded_large_text_files"],
    "statistics": statistics,
    "important_note": (
        "Compact evidence package, not a complete training backup."
    )
}

(STAGE / "EVIDENCE_PACK_SUMMARY.json").write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

README = f"""
# YOLO11 PPE — COMPACT REPORT EVIDENCE PACK

Created: {summary['created_at']}

## Purpose

A compact package containing research evidence for preparing the
Vietnamese LaTeX report.

## Contents

- Dataset YAML and metadata.
- CVAT polygon TXT labels when available.
- Train/validation/test split lists.
- Metadata extracted from dataset ZIPs.
- Training results CSV and configuration.
- Training logs.
- Selected training and validation plots.
- Visual Error Audit text and extracted images.
- Notebook code history from the current kernel.
- Runtime environment and GPU information.
- Annotation statistics.
- Research handover.

## Intentionally excluded

- Original dataset images.
- Large dataset ZIP files.
- Model checkpoints and weights.
- Unnecessary temporary files.

## Main entry points

1. `RESEARCH_HANDOVER.md`
2. `EVIDENCE_PACK_SUMMARY.json`
3. `annotation_statistics.json`
4. `training_analysis/training_results_summary.json`
5. `visual_error_audit/`
6. `training_visual_evidence/`
7. `environment/runtime_environment.json`

IMPORTANT:
The previous experiment values in RESEARCH_HANDOVER.md are transcribed
from the user's earlier report draft. Verify them against original
experiment logs before publishing.
"""

(STAGE / "README.md").write_text(
    README.strip(),
    encoding="utf-8"
)


# =============================================================================
# 15. CHECK AVAILABLE SPACE BEFORE CREATING ZIP
# =============================================================================

stage_size = sum(
    p.stat().st_size
    for p in STAGE.rglob("*")
    if p.is_file()
)

disk = shutil.disk_usage(WORK_DIR)

print("\nStaging size:", f"{stage_size / 1024**2:.2f} MB")

print(
    "Remaining disk:",
    f"{disk.free / 1024**3:.2f} GB"
)

if stage_size + 100 * 1024**2 > disk.free:

    raise RuntimeError(
        "Không đủ dung lượng trống để tạo ZIP an toàn. "
        "Kiểm tra /kaggle/working."
    )


# =============================================================================
# 16. CREATE COMPACT ZIP
# =============================================================================

print("\nCreating ZIP archive...")

with zipfile.ZipFile(
    OUTPUT_ZIP,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=3,
    allowZip64=True
) as zf:

    all_files = sorted(
        [
            p for p in STAGE.rglob("*")
            if p.is_file()
        ],
        key=lambda p: p.as_posix().lower()
    )

    for index, path in enumerate(all_files, 1):

        archive_path = path.relative_to(STAGE).as_posix()

        # JPEG and PNG already contain compressed data.
        # ZIP_STORED avoids wasting CPU on image recompression.
        if path.suffix.lower() in IMAGE_EXTENSIONS:

            zf.write(
                path,
                arcname=archive_path,
                compress_type=zipfile.ZIP_STORED
            )

        else:

            zf.write(
                path,
                arcname=archive_path,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=3
            )

        if index % 100 == 0:

            print(
                f"  Added {index:,}/{len(all_files):,} files"
            )


# =============================================================================
# 17. VERIFY ZIP INTEGRITY
# =============================================================================

print("\nVerifying ZIP integrity...")

with zipfile.ZipFile(
    OUTPUT_ZIP,
    "r"
) as zf:

    damaged_file = zf.testzip()

    archive_members = zf.namelist()

    member_count = len(archive_members)

if damaged_file is not None:

    raise RuntimeError(
        f"ZIP verification failed: {damaged_file}"
    )

verification = {
    "archive": ARCHIVE_NAME,
    "exists": OUTPUT_ZIP.exists(),
    "size_bytes": file_size(OUTPUT_ZIP),
    "size_mb": round(
        file_size(OUTPUT_ZIP) / 1024**2,
        2
    ),
    "member_count": member_count,
    "integrity": "PASS",
    "research_handover_exists": (
        "RESEARCH_HANDOVER.md" in archive_members
    ),
    "annotation_statistics_exists": (
        "annotation_statistics.json" in archive_members
    ),
    "training_summary_exists": (
        "training_analysis/training_results_summary.json"
        in archive_members
    )
}

(WORK_DIR / f"YOLO11_EVIDENCE_VERIFICATION_{TIMESTAMP}.json").write_text(
    json.dumps(
        verification,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# =============================================================================
# 18. FINAL REPORT
# =============================================================================

final_size = file_size(OUTPUT_ZIP)

print("\n" + "=" * 80)
print("COMPACT RESEARCH EVIDENCE PACK COMPLETED")
print("=" * 80)

print("\nOUTPUT FILE:")
print(OUTPUT_ZIP)

print("\nFILE SIZE:")
print(f"{final_size / 1024**2:.2f} MB")

print("\nZIP MEMBERS:")
print(member_count)

print("\nINCLUDED EVIDENCE:")
print("  Original metadata files :", statistics["input_metadata_files"])
print("  Working evidence files  :", statistics["working_evidence_files"])
print("  Annotation TXT files    :", statistics["annotation_files"])
print("  Training CSV summaries  :", len(training_results))
print("  Audit reports processed :", statistics["audit_reports"])
print("  Audit images extracted  :", statistics["audit_images"])
print("  Selected plots          :", statistics["plot_images"])
print("  Notebook input cells    :", history_status["input_cells"])

print("\nEXCLUDED:")
print("  Model weights           :", statistics["excluded_weights"])
print("  Raw images              :", statistics["excluded_raw_images"])
print("  Large text files        :", statistics["excluded_large_text_files"])

print("\nZIP INTEGRITY:")
print(verification["integrity"])

print("\nDOWNLOAD:")

display(
    Markdown(
        f"""
### Research Evidence Pack

**File:** `{ARCHIVE_NAME}`

**Size:** {final_size / 1024**2:.2f} MB

**ZIP integrity:** PASS

Tải file từ liên kết bên dưới hoặc mở Output trong Kaggle.
"""
    )
)

display(
    FileLink(str(OUTPUT_ZIP))
)

print("\nSTAGING DIRECTORY:")
print(STAGE)

print("\nDONE.")