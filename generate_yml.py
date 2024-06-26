import re

# Read requirements.txt
with open('requirements.txt', 'r') as req_file:
    lines = req_file.readlines()

# Extract package names and versions
dependencies = []
pip_dependencies = []
for line in lines:
    line = line.strip()
    if line and not line.startswith("#"):
        if line.startswith("-e") or "==" in line:
            pip_dependencies.append(line)
        else:
            dependencies.append(line)

# Write environment.yml
with open('environment.yml', 'w') as yml_file:
    yml_file.write("name: creditfraud\n")
    yml_file.write("channels:\n")
    yml_file.write("  - conda-forge\n")  # Give conda-forge higher priority
    yml_file.write("  - defaults\n")
    yml_file.write("dependencies:\n")
    yml_file.write("  - python>=3.12.3\n")
    for dependency in dependencies:
        yml_file.write(f"  - {dependency}\n")
    yml_file.write("  - pip\n")
    yml_file.write("  - pip:\n")
    for dependency in pip_dependencies:
        yml_file.write(f"    - {dependency}\n")

print("environment.yml has been generated successfully.")
