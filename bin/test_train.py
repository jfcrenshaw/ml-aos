"""Run tests to validate network setup and training."""
import pytorch_lightning as pl

from ml_aos.lightning import DonutLoader, WaveNetSystem
from ml_aos.utils import printOnce

# first we will do a fast dev run to make sure no errors are thrown
printOnce("fast dev run", header=True)
pl.seed_everything(42, workers=True)
trainer = pl.Trainer(fast_dev_run=True, deterministic=True)
trainer.fit(model=WaveNetSystem(), train_dataloaders=DonutLoader())

# next we will overfit a single batch
printOnce("overfit single batch", header=True)
pl.seed_everything(42, workers=True)
trainer = pl.Trainer(
    overfit_batches=1,
    log_every_n_steps=1,
    max_epochs=100,
    deterministic=True,
    callbacks=[pl.callbacks.EarlyStopping("val_loss", patience=10)],
    logger=pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name="overfit_test"),
)
trainer.fit(model=WaveNetSystem(), train_dataloaders=DonutLoader())
