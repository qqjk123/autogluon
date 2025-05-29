# run_classifier.py
from autogluon.multimodal.learners import MultiModalClassificationLearner

if __name__ == "__main__":
    learner = MultiModalClassificationLearner(
        #save_dir="/workspace/multimodal_classifier/dataset100",
        #metadata_csv="/workspace/Dataset100_CLS/metadata.csv",
        save_dir="/workspace/multimodal_classifier/dataset020",
        metadata_csv="/workspace/Dataset020_BraTS2021_CLS/metadata.csv",
        modalities=["FLAIR", "T1w", "T1wCE", "T2w"],
        img_size=(128, 128, 128),
        lr=1e-4,
        batch_size=8,
        val_split=0.2,
        epochs=2,
        num_workers=50,
        n_bootstrap=1000,
        ci_alpha=0.05,
        device="cuda"
    )

    learner.fit()
    learner.evaluate()
