"""Train the WaveNet."""
import pytorch_lightning as pl

from ml_aos.lightning import DonutLoader, WaveNetSystem

# create callbacks
early_stopping = pl.callbacks.EarlyStopping("val_loss", patience=20)
val_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")

# seed everything so we're deterministic
pl.seed_everything(42, workers=True)

# create the model, dataloader, and trainer
# model = WaveNetSystem()
# dataloader = DonutLoader()
trainer = pl.Trainer(
    deterministic=True,
    precision=16,
    devices=2,
    accelerator="gpu",
    callbacks=[
        early_stopping,
        val_checkpoint,
    ],
    log_every_n_steps=50,
)

# search for a good learning rate
# tuner = pl.tuner.tuning.Tuner(trainer)
# lr_finder = tuner.lr_find(model, datamodule=dataloader)
# model.hparams.lr = lr_finder.suggestion()

# fit!
trainer.fit(
    model=WaveNetSystem(
        lr=0.002,
        n_meta_layers=3,
        n_meta_nodes=128,
        n_predictor_layers=(256, 128, 64, 32),
    ),
    train_dataloaders=DonutLoader(),
)
