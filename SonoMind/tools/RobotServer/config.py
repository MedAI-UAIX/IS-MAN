import os

# Storage path for subsequent analysis of the article
project_root = os.path.dirname(os.path.abspath(__file__))
record_save_path = os.path.join(project_root, 'data', 'for_paper')  # Relative Path
os.makedirs(record_save_path, exist_ok=True)  # Automatically create directory