import os

def list_mobile_sam_tree(root_path, indent=0):
    """Affiche la structure hiérarchique du dossier MobileSAM."""
    prefix = "│   " * (indent - 1) + ("├── " if indent > 0 else "")
    if indent == 0:
        print(f"📂 {os.path.basename(root_path)}")
    else:
        print(f"{prefix}{os.path.basename(root_path)}/")

    try:
        entries = sorted(os.listdir(root_path))
    except PermissionError:
        print(prefix + "🚫 [Permission Denied]")
        return

    for entry in entries:
        path = os.path.join(root_path, entry)
        if os.path.isdir(path):
            list_mobile_sam_tree(path, indent + 1)
        else:
            print("│   " * indent + f"├── {entry}")

# === Chemin du dossier source MobileSAM ===
root_dir = r"C:\Users\maxam\Desktop\TM\MobileSAM\mobile_sam"

if __name__ == "__main__":
    if os.path.exists(root_dir):
        list_mobile_sam_tree(root_dir)
    else:
        print(f"❌ Le chemin n'existe pas : {root_dir}")
