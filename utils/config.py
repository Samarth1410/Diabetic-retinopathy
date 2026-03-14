import os

class Config:
    # ── Paths ──────────────────────────────────────────────────
    BASE_DIR   = os.path.join(os.path.expanduser("~"), "diabetic-retinopathy")
    DATA_DIR   = os.path.join(BASE_DIR, "data", "aptos2019")
    TRAIN_CSV  = os.path.join(DATA_DIR, "train.csv")
    TRAIN_IMGS = os.path.join(DATA_DIR, "train_images")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    # ── Sampling ───────────────────────────────────────────────
    MAX_PER_CLASS = 500        # cap per class; set None for full dataset

    # ── Image ──────────────────────────────────────────────────
    IMG_SIZE = 224
    MEAN     = [0.485, 0.456, 0.406]   # ImageNet stats
    STD      = [0.229, 0.224, 0.225]

    # ── Training ───────────────────────────────────────────────
    BATCH_SIZE      = 32
    NUM_EPOCHS      = 20
    LR              = 1e-4
    LR_BACKBONE     = 1e-5
    WEIGHT_DECAY    = 1e-4
    UNFREEZE_EPOCH  = 5        # epoch at which Block5 of VGG16 is unfrozen

    # ── Model ──────────────────────────────────────────────────
    NUM_CLASSES = 5
    DROPOUT     = 0.5

    # ── Misc ───────────────────────────────────────────────────
    SEED        = 42
    NUM_WORKERS = 2
    SAVE_PATH   = os.path.join(OUTPUT_DIR, "best_model.pth")

    # ── Labels ─────────────────────────────────────────────────
    CLASS_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    CLASS_COLORS = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]
