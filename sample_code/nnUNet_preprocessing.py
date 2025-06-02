#6 MET
# Update dataset.json
import os
import json

# === 用户需要修改的部分 ===
# 指向 nnUNet_raw 下的 Dataset 文件夹
dataset_root = "/workspace/nnUNet_raw/Dataset006_BraTS2023_MET"

# dataset.json 中的 “name”和 “description”
dataset_name = "Dataset006_BraTS2023_MET"
description = "nnU-Net dataset 6: BraTS2023_MET"

# 通道映射（必须是字符串键："0","1","2","3" → 对应模态名称）
channel_names = {
    "0": "T1C",
    "1": "T1N",
    "2": "T2F",
    "3": "T2W"
}

# 文件后缀
file_ending = ".nii.gz"

# “labels” 字段必须包含 "background": 0，且其他类别从 1 开始连续
labels_mapping = {
    "background": 0,
    "1": 1,
    "2": 2,
    "3": 3
}
# ==================================

# 拼接子文件夹路径
imagesTr_dir = os.path.join(dataset_root, "imagesTr")
labelsTr_dir = os.path.join(dataset_root, "labelsTr")
imagesTs_dir = os.path.join(dataset_root, "imagesTs")

# ------------------------------------------------------------------------------
# 1. 构造 training 列表：跳过隐藏文件，确保每个 case 的所有通道图像和对应 label 都存在
# ------------------------------------------------------------------------------
training_list = []
seen_cases = set()

for fname in os.listdir(imagesTr_dir):
    # 跳过以 '.' 开头的隐藏文件（如 .DS_Store 或 ._xxx）
    if fname.startswith("."):
        continue
    # 只匹配通道 0 文件名："<case_id>_0000.nii.gz"
    if not fname.endswith(f"_0000{file_ending}"):
        continue

    case_id = fname.replace(f"_0000{file_ending}", "")
    if case_id.startswith(".") or case_id in seen_cases:
        continue
    seen_cases.add(case_id)

    # 检查对应的 label 是否存在
    label_path = os.path.join(labelsTr_dir, case_id + file_ending)
    if not os.path.isfile(label_path):
        print(f"跳过（缺少 label 文件）：{case_id}")
        continue

    # 检查所有通道图像是否都存在
    image_files = []
    ok = True
    for ch_key in sorted(channel_names, key=lambda x: int(x)):
        img_name = f"{case_id}_{int(ch_key):04d}{file_ending}"
        img_path = os.path.join(imagesTr_dir, img_name)
        if not os.path.isfile(img_path):
            print(f"跳过（缺少通道图像）：{case_id}，缺少 {img_name}")
            ok = False
            break
        image_files.append(img_name)
    if not ok:
        continue

    training_list.append({
        "image": image_files,
        "label": case_id + file_ending
    })

# ------------------------------------------------------------------------------
# 2. 构造 test 列表（若 imagesTs 存在），同样跳过隐藏文件并确保每个 case 的所有通道都存在
# ------------------------------------------------------------------------------
test_list = []
if os.path.isdir(imagesTs_dir):
    seen_test = set()
    for fname in os.listdir(imagesTs_dir):
        if fname.startswith("."):
            continue
        if not fname.endswith(f"_0000{file_ending}"):
            continue

        case_id = fname.replace(f"_0000{file_ending}", "")
        if case_id.startswith(".") or case_id in seen_test:
            continue
        seen_test.add(case_id)

        image_files = []
        ok = True
        for ch_key in sorted(channel_names, key=lambda x: int(x)):
            img_name = f"{case_id}_{int(ch_key):04d}{file_ending}"
            img_path = os.path.join(imagesTs_dir, img_name)
            if not os.path.isfile(img_path):
                print(f"跳过测试集（缺少通道图像）：{case_id}，缺少 {img_name}")
                ok = False
                break
            image_files.append(img_name)
        if not ok:
            continue

        test_list.append({ "image": image_files })

# ------------------------------------------------------------------------------
# 3. 合成 dataset.json 内容
# ------------------------------------------------------------------------------
dataset_json = {
    "name": dataset_name,
    "description": description,
    "tensorImageSize": "4D",
    "channel_names": channel_names,
    "file_ending": file_ending,
    "labels": labels_mapping,
    "numTraining": len(training_list),
    "numTest": len(test_list),
    "training": training_list
}
if test_list:
    dataset_json["test"] = test_list

# ------------------------------------------------------------------------------
# 4. 写入到 dataset.json（覆盖原文件）
# ------------------------------------------------------------------------------
out_path = os.path.join(dataset_root, "dataset.json")
with open(out_path, "w") as f:
    json.dump(dataset_json, f, indent=4, ensure_ascii=False)

print(f"✓ 已生成并保存：{out_path}")
print(f"共计 {len(training_list)} 个训练 case，{len(test_list)} 个测试 case。")








---------------------------------------------------------------------------------------------------
# Regenerate Dataset.json
# Update dataset.json
import os
import json

# === 用户需要修改的部分 ===
# 指向 nnUNet_raw 下的 Dataset 文件夹
dataset_root = "/workspace/nnUNet_raw/Dataset002_BraTS2023_MEN"

# dataset.json 中的 “name”和 “description”
dataset_name = "Dataset002_BraTS2023_MEN"
description = "nnU-Net dataset 2: BraTS2023_MEN"

# 通道映射（必须是字符串键："0","1","2","3" → 对应模态名称）
channel_names = {
    "0": "T1C",
    "1": "T1N",
    "2": "T2F",
    "3": "T2W"
}

# 文件后缀
file_ending = ".nii.gz"

# “labels” 字段必须包含 "background": 0，且其他类别从 1 开始连续
labels_mapping = {
    "background": 0,
    "1": 1,
    "2": 2,
    "3": 3
}
# ==================================

# 拼接子文件夹路径
imagesTr_dir = os.path.join(dataset_root, "imagesTr")
labelsTr_dir = os.path.join(dataset_root, "labelsTr")
imagesTs_dir = os.path.join(dataset_root, "imagesTs")

# ------------------------------------------------------------------------------
# 1. 构造 training 列表：跳过隐藏文件，确保每个 case 的所有通道图像和对应 label 都存在
# ------------------------------------------------------------------------------
training_list = []
seen_cases = set()

for fname in os.listdir(imagesTr_dir):
    # 跳过以 '.' 开头的隐藏文件（如 .DS_Store 或 ._xxx）
    if fname.startswith("."):
        continue
    # 只匹配通道 0 文件名："<case_id>_0000.nii.gz"
    if not fname.endswith(f"_0000{file_ending}"):
        continue

    case_id = fname.replace(f"_0000{file_ending}", "")
    if case_id.startswith(".") or case_id in seen_cases:
        continue
    seen_cases.add(case_id)

    # 检查对应的 label 是否存在
    label_path = os.path.join(labelsTr_dir, case_id + file_ending)
    if not os.path.isfile(label_path):
        print(f"跳过（缺少 label 文件）：{case_id}")
        continue

    # 检查所有通道图像是否都存在
    image_files = []
    ok = True
    for ch_key in sorted(channel_names, key=lambda x: int(x)):
        img_name = f"{case_id}_{int(ch_key):04d}{file_ending}"
        img_path = os.path.join(imagesTr_dir, img_name)
        if not os.path.isfile(img_path):
            print(f"跳过（缺少通道图像）：{case_id}，缺少 {img_name}")
            ok = False
            break
        image_files.append(img_name)
    if not ok:
        continue

    training_list.append({
        "image": image_files,
        "label": case_id + file_ending
    })

# ------------------------------------------------------------------------------
# 2. 构造 test 列表（若 imagesTs 存在），同样跳过隐藏文件并确保每个 case 的所有通道都存在
# ------------------------------------------------------------------------------
test_list = []
if os.path.isdir(imagesTs_dir):
    seen_test = set()
    for fname in os.listdir(imagesTs_dir):
        if fname.startswith("."):
            continue
        if not fname.endswith(f"_0000{file_ending}"):
            continue

        case_id = fname.replace(f"_0000{file_ending}", "")
        if case_id.startswith(".") or case_id in seen_test:
            continue
        seen_test.add(case_id)

        image_files = []
        ok = True
        for ch_key in sorted(channel_names, key=lambda x: int(x)):
            img_name = f"{case_id}_{int(ch_key):04d}{file_ending}"
            img_path = os.path.join(imagesTs_dir, img_name)
            if not os.path.isfile(img_path):
                print(f"跳过测试集（缺少通道图像）：{case_id}，缺少 {img_name}")
                ok = False
                break
            image_files.append(img_name)
        if not ok:
            continue

        test_list.append({ "image": image_files })

# ------------------------------------------------------------------------------
# 3. 合成 dataset.json 内容
# ------------------------------------------------------------------------------
dataset_json = {
    "name": dataset_name,
    "description": description,
    "tensorImageSize": "4D",
    "channel_names": channel_names,
    "file_ending": file_ending,
    "labels": labels_mapping,
    "numTraining": len(training_list),
    "numTest": len(test_list),
    "training": training_list
}
if test_list:
    dataset_json["test"] = test_list

# ------------------------------------------------------------------------------
# 4. 写入到 dataset.json（覆盖原文件）
# ------------------------------------------------------------------------------
out_path = os.path.join(dataset_root, "dataset.json")
with open(out_path, "w") as f:
    json.dump(dataset_json, f, indent=4, ensure_ascii=False)

print(f"✓ 已生成并保存：{out_path}")
print(f"共计 {len(training_list)} 个训练 case，{len(test_list)} 个测试 case。")





 ------------------------------------------------------------------------------ ------------------------------------------------------------------------------



# Delete hidden files which start with . 


import os

def delete_dot_files(root_dir):
    """
    递归删除 root_dir 及其子目录下所有以 '.' 开头的文件（隐藏文件）。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.startswith('.'):
                full_path = os.path.join(dirpath, fname)
                try:
                    os.remove(full_path)
                    print(f"Deleted: {full_path}")
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")

if __name__ == "__main__":
    # 将下面路径修改为你想要清理的根目录
    target_directory = "/workspace/nnUNet_raw/Dataset006_BraTS2023_MET"
    delete_dot_files(target_directory)












