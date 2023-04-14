"""Export the pytorch lightning checkpoint to a torchscript model."""
from datetime import datetime
from zoneinfo import ZoneInfo

import click
import torch

from ml_aos.lightning import WaveNetSystem
from ml_aos.utils import get_root


@click.command()
@click.option(
    "--version",
    "-v",
    help=(
        "Number of model version to export. "
        "Must correspond to one of the subdirectories in lightning_logs."
    ),
    required=True,
    type=int,
)
@click.option(
    "--name",
    default=None,
    help=(
        "The model is saved as models/{name}.pt. "
        "If None, then the version + current timestamp is used."
    ),
    type=str,
)
def export_model(version: str, name: str) -> None:
    """Export the model to torchscript and save in the models directory."""
    # get the root of the git repo
    root = get_root()

    # get the path of the checkpoint file
    ckpt_dir = root / "lightning_logs" / f"version_{version}" / "checkpoints"
    assert ckpt_dir.exists(), f"directory {ckpt_dir} does not exist."
    ckpt = list(ckpt_dir.glob("*"))[0]

    # load the checkpoint and convert it to torchscript
    script = WaveNetSystem.load_from_checkpoint(ckpt).to_torchscript()

    # get the output file path
    if name is None:
        time_stamp = datetime.now(ZoneInfo("America/Los_Angeles"))
        name = f"v{version}_" + str(time_stamp).split(".")[0].replace(" ", "_")
    out_path = root / "models" / (name + ".pt")

    # save the torchscript model
    torch.jit.save(script, out_path)


if __name__ == "__main__":
    export_model()
