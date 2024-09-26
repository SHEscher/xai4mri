# Installation of `xai4mri`

## Python versions

`xai4mri` requires `Python 3.9` or later, but smaller than `<3.12`.

## Pre-requisites

As a first step, it is always recommended to create a new virtual environment.

`xai4mri` is specifically developed for MRI-based research.
Check out the [`scilaunch`](https://shescher.github.io/scilaunch/) package,
which helps to set up new research projects
including virtual environments, data management, and more.

Virtual environments can also be creating using `venv`:

```shell
python -m venv "myenv_3.11" --python=python3.11
source myenv_3.11/bin/activate
```

Or using [`conda`](https://docs.anaconda.com/miniconda/):

```shell
conda create -n "myenv_3.11" python=3.11
conda activate myenv_3.11
```

## Installation

To install the package, simply use `pip`:

```shell
pip install -U xai4mri
```

![brain](assets/images/favicon.ico)

You are now ready to use `xai4mri` in your Python environment.
