"""Run tests to validate network setup and training."""
import pytorch_lightning as pl

from ml_aos.lightning import DonutLoader, WaveNetSystem
from ml_aos.utils import printOnce

# first we will do a fast dev run to make sure no errors are thrown
printOnce("fast dev run", header=True)
pl.seed_everything(42, workers=True)
trainer = pl.Trainer(fast_dev_run=True, deterministic=True)
trainer.fit(model=WaveNetSystem(), train_dataloaders=DonutLoader(shuffle=False))

# next we will overfit a single batch
printOnce("overfit single batch", header=True)
pl.seed_everything(42, workers=True)
trainer = pl.Trainer(
    overfit_batches=1,
    log_every_n_steps=1,
    max_epochs=400,
    deterministic=True,
    callbacks=[
        pl.callbacks.EarlyStopping("train_loss", patience=20),
        pl.callbacks.LearningRateMonitor(),
    ],
    logger=pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name="overfit_test"),
)
trainer.fit(
    model=WaveNetSystem(lr=1e-2, n_meta_layers=2, n_meta_nodes=32),
    train_dataloaders=DonutLoader(batch_size=64, shuffle=False),
)
