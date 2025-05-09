import os
import nibabel as nib
import numpy as np

def inspect_label_values(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    vals = np.unique(data.astype(np.int32))
    print(f"{os.path.basename(nifti_path)} → labels: {vals}")

def inspect_all_labels(dataset_root):
    labels_dir = os.path.join(dataset_root, "labelsTr")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"找不到 labelsTr 目录: {labels_dir}")
    for fname in sorted(os.listdir(labels_dir)):
        # 跳过隐藏文件
        if fname.startswith("."):
            continue
        if fname.endswith(".nii.gz"):
            inspect_label_values(os.path.join(labels_dir, fname))

if __name__ == "__main__":
    dataset_root = "/home/ubuntu/nnUNet_raw/Dataset002_BraTS2023_SEG"
    inspect_all_labels(dataset_root)
