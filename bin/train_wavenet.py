"""Train the WaveNet."""
import pytorch_lightning as pl

from ml_aos.lightning import DonutLoader, WaveNetSystem

# create callbacks
early_stopping = pl.callbacks.EarlyStopping("val_loss", patience=20)
val_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
lr_monitor = pl.callbacks.LearningRateMonitor()

# seed everything so we're deterministic
pl.seed_everything(42, workers=True)

# create the model, dataloader, and trainer
model = WaveNetSystem(
    lr=0.0003,
    n_predictor_layers=(171, 57),
    lr_schedule=True,
    alpha=0,
)

data_loader = DonutLoader()

trainer = pl.Trainer(
    deterministic=True,
    precision="16-mixed",
    devices=1,
    accelerator="gpu",
    callbacks=[
        early_stopping,
        val_checkpoint,
        lr_monitor,
    ],
    log_every_n_steps=50,
    max_epochs=-1,
)

# fit!
trainer.fit(model, data_loader)
