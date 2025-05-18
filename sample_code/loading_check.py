import json
from pathlib import Path
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, Lambdad, ScaleIntensityRanged, DivisiblePadd, ToTensord

# 1. Load your dataset.json
ds = Path("/workspace/nnUnet_raw") / "Dataset006_BraTS2023_MET"
info = json.load(open(ds / "dataset.json"))
examples = info["training"]

# 2. Build the exact same transform pipeline you use in CacheDataset
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Lambdad(keys="label", func=lambda x: x),            # or your LABEL_MAP logic
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
    DivisiblePadd(keys=["image", "label"], k=8),
    ToTensord(keys=["image", "label"]),
])

# 3. Loop and catch errors
for idx, ex in enumerate(examples):
    # build a single-data dict exactly as CacheDataset would see it
    data_dict = {
        "image": [str(ds/"imagesTr"/fn) for fn in ex["image"]],
        "label": str(ds/"labelsTr"/ex["label"])
    }
    try:
        out = transforms(data_dict)  # this runs all your transforms
    except Exception as e:
        print(f"❌ Error at index {idx} →")
        print("   image files:", data_dict["image"])
        print("   label file:", data_dict["label"])
        print("   caught exception:", repr(e))
        break
else:
    print("✅ All samples passed the MONAI transforms!")
