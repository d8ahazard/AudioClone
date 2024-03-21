import os
import pkgutil
import sys
import pkg_resources

# List all standard library modules
std_lib = {name for _, name, _ in pkgutil.iter_modules() if name in sys.builtin_module_names}

all_files = []
# Enumerate all of the scripts in the current directory
for root, dirs, files in os.walk("."):
    for file in files:
        full_path = os.path.join(root, file)
        # Skip files that are not Python scripts, or are within a virtual environment directory
        if file.endswith(".py") and not file.startswith("__") and "venv" not in full_path:
            all_files.append(full_path)

all_imports = set()
for file in all_files:
    with open(file, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("from") or stripped_line.startswith("import"):
                # Simplify the parsing by splitting on 'import' and 'from', then taking the first part
                parts = stripped_line.replace("import ", ",").replace("from ", ",").split(",")
                if "as" in parts[1]:
                    base_module = parts[1].split("as")[0].strip()
                else:
                    base_module = parts[1].split(".")[0].strip()
                if base_module:
                    all_imports.add(base_module.lower())

print(f"All Imports: {all_imports}")
# Filter out standard library imports and non-installed packages
non_std_imports = {imp for imp in all_imports if imp not in std_lib}

# Get installed versions for non-standard imports
non_std_packages_with_versions = {pkg.key: pkg.version for pkg in pkg_resources.working_set if pkg.key in non_std_imports}

# Print non-standard imports with versions
for pkg, version in sorted(non_std_packages_with_versions.items()):
    version = version.replace("+cu121", "")
    print(f"{pkg}=={version}")
print("--extra-index-url https://download.pytorch.org/whl/cu121")