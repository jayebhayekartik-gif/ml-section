import os

# =============== FOLDER STRUCTURE =================
structure = {
    "data": ["raw", "processed", "external"],
    "notebooks": [],
    "src": [
        "data_preprocessing",
        "features",
        "models",
        "utils"
    ],
    "experiments": [],
    "models": [],
    "reports": ["figures"],
    "tests": []
}

files_to_create = [
    "README.md",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    "experiments/experiment_log.md",
    "reports/report.md",
    "src/__init__.py",
    "src/data_preprocessing/preprocess.py",
    "src/features/build_features.py",
    "src/models/train.py",
    "src/models/evaluate.py",
    "src/models/predict.py",
    "src/utils/helpers.py",
    "tests/test_preprocess.py",
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_model_training.ipynb",
]

# =============== CREATE FOLDERS =================
for folder, subfolders in structure.items():
    os.makedirs(folder, exist_ok=True)
    for sub in subfolders:
        os.makedirs(os.path.join(folder, sub), exist_ok=True)

# =============== CREATE FILES =================
for file in files_to_create:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            if file == "README.md":
                f.write("# Smart Pricing Intelligence Platform - Machine Learning\n")
                f.write("This repo contains the ML pipeline for dynamic pricing intelligence.\n")
            elif file == "requirements.txt":
                f.write("# Add Python dependencies here\npandas\nnumpy\nscikit-learn\nmatplotlib\njupyter\n")
            elif file == ".gitignore":
                f.write(
"""# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.so
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
.notebooks/

# Virtual env
.env/
.venv/
venv/
ENV/

# Data
data/raw/*
data/processed/*
!data/.gitkeep

# Models
models/*
!models/.gitkeep

# Logs
*.log
""")
            elif file.endswith(".py"):
                f.write("# " + file.split("/")[-1] + "\n")
            elif file.endswith(".md"):
                f.write("# " + file.split("/")[-1].replace(".md", "") + "\n")

# Add placeholder files for empty folders
open("data/.gitkeep", "w").close()
open("models/.gitkeep", "w").close()

print("✅ Repository structure created successfully!")
