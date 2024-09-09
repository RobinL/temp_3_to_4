import json
import os
import shutil

import nbformat


def extract_python_code(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    python_code = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            python_code.append(cell.source)

    return "\n\n".join(python_code)


def process_notebooks(source_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".ipynb") and "checkpoint" not in file:
                notebook_path = os.path.join(root, file)
                python_code = extract_python_code(notebook_path)

                output_filename = os.path.splitext(file)[0] + ".py"
                output_path = os.path.join(target_dir, output_filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(python_code)


# Process Splink 3 notebooks
process_notebooks("splink3_ipynbs", "splink3_flat")

# Process Splink 4 notebooks
process_notebooks("splink4_ipynbs", "splink4_flat")
