import os
import shutil
import json
from pathlib import Path

def convert_to_nnunet(input_root: str,
                     nnunet_raw: str,
                     dataset_id: int,
                     dataset_name: str,
                     num_cases: int = None):
    """
    转换 BraTS 风格的病例文件夹到 nnU‑Net raw 格式。

    Args:
        input_root:   原始病例根目录，含子文件夹每个病例
        nnunet_raw:   输出 nnUNet_raw 目录
        dataset_id:   数据集编号（整数），用于创建 DatasetXXX_{name}
        dataset_name: 数据集名称（字符串）
        num_cases:    要包含的病例总数；None 表示包含所有病例
    """
    ds_key = str(dataset_id).zfill(3) + '_' + dataset_name
    out_base = Path(nnunet_raw) / f'Dataset{ds_key}'
    imagesTr = out_base / 'imagesTr'
    labelsTr = out_base / 'labelsTr'
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    # 定义模态顺序：0=seg,1=t1c,2=t1n,3=t2f,4=t2w
    mod_keys = ['seg', 't1c', 't1n', 't2f', 't2w']

    # 获取所有病例文件夹
    cases = [d for d in sorted(os.listdir(input_root)) if (Path(input_root)/d).is_dir()]
    if num_cases is not None:
        cases = cases[:num_cases]

    training = []
    for case in cases:
        case_dir = Path(input_root) / case
        # 收集各模态文件
        files = { key: None for key in mod_keys }
        for f in case_dir.iterdir():
            name = f.name.lower()
            if name.startswith('._'):
                continue
            for key in mod_keys:
                if key in name and name.endswith('.nii.gz'):
                    files[key] = f
        missing = [k for k,v in files.items() if v is None]
        if missing:
            print(f"⚠️ 跳过 {case}：缺少模态 {missing}")
            continue

        # 复制并重命名到 imagesTr
        for idx, key in enumerate(mod_keys):
            src = files[key]
            dst_name = f"{case}_{idx:04d}.nii.gz"
            shutil.copy(src, imagesTr / dst_name)
        # 复制 seg 到 labelsTr
        shutil.copy(files['seg'], labelsTr / f"{case}.nii.gz")

        training.append({
            "image": [ f"{case}_{i:04d}.nii.gz" for i in range(len(mod_keys)) ],
            "label": f"{case}.nii.gz"
        })
        print(f"✔️ 处理病例 {case}")

    # 写 dataset.json
    dataset_json = {
        "name": dataset_name,
        "description": f"nnU-Net dataset {dataset_id}: {dataset_name}",
        "tensorImageSize": "4D",
        "reference": "",
        "license": "",
        "release": "1.0",
        "modality": { str(i): k.upper() for i,k in enumerate(mod_keys) },
        "labels": {"0": "background", "1": "tumor"},
        "numTraining": len(training),
        "numTest": 0,
        "training": training,
        "test": []
    }
    with open(out_base / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\n已创建 nnU-Net 数据集：{out_base}")

if __name__ == "__main__":
    # 示例：只处理前 10 个病例
    convert_to_nnunet(
        input_root="/Volumes/ssd/BraTS-MEN-Train",
        nnunet_raw="/Volumes/ssd/nnUNet_raw",
        dataset_id=1,
        dataset_name="BraTS2023_SEG",
        num_cases=10
    )
