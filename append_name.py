# append_name.py
import os

# Define root path
project_root = r"D:\git_projects\dynamic_fraud_detection"
src_dir = os.path.join(project_root)

# Collect all .py files in root and src (recursively)
target_files = []

# Add .py files in root
for file in os.listdir(project_root):
    if file.endswith(".py"):
        target_files.append(os.path.join(project_root, file))

# Add .py files recursively in src/
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".py"):
            target_files.append(os.path.join(root, file))

# Process each .py file
for file_path in target_files:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filename = os.path.basename(file_path)
    comment_line = f"# {filename}\n"

    # If first line is not the expected comment, prepend it
    if not lines or lines[0].strip() != f"# {filename}":
        print(f"Updating: {filename}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(comment_line)
            f.writelines(lines)
    else:
        print(f"Already OK: {filename}")

print("\n✅ Done: All Python files now start with a filename comment.")
