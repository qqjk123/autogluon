# start_training.py

from autogluon.multimodal.learners import ResNetPredictor

def main():
    trainer = ResNetPredictor(
        metadata_csv = "/workspace/Dataset020_BraTS2021_CLS/metadata.csv",
        output_dir = "/workspace/resnetbaseline/dataset020",
        #metadata_csv = "/workspace/Dataset100_CLS/metadata.csv",
        #output_dir = "/workspace/resnetbaseline/dataset100",
        in_channels=4,
        batch_size=2,
        lr=2e-4,
        weight_decay=1e-5,
        max_epochs=100,
        patience=10,
        accumulation_steps=1,
        use_amp=True,
        val_split=0.2,
        num_workers=20,
        pin_memory=True,
        n_bootstrap=1000,
        ci_alpha=0.05
    )
    trainer.fit()
    # 若需在训练结束后自动评估，可在此启用：
    trainer.evaluate()

if __name__ == "__main__":
    main()
