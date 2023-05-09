# ml-aos

Machine Learning methods for Rubin AOS system

### Installation

1. Make sure Poetry and Conda/Mamba are installed.
2. Create a new conda environment, activate it, and run `poetry install`
3. If developing, install the pre-commit hooks: `pre-commit install`

### Running the code

Training and exporting models is handled by the scripts in `bin/`.

- `test_train.py` - runs quick sanity checks to make sure the model is set up correctly
- `train_wavenet.py` - trains the network
- `export_model.py` - exports the model in torchscript format

### Notebooks

Some basic analysis is present in `notebooks/`.
More analysis is in the [aos_notebooks](https://github.com/jfcrenshaw/aos_notebooks) repo.
