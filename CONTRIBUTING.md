# Contributing

Contributions are very welcome, and credit will always be given.

## Setup for development

Fork & then clone the repo:

Install the dependencies for development,
ideally in a virtual environment in editable mode:

```shell
cd xai4mri
pip install -e ".[develop,docs]"
```

## Development workflow

Make changes on a new branch:

```shell
git checkout -b develop
```

### Testing

Run the tests:

```shell
pytest . --cov --cov-report=html
```

### Documentation

Use [`sphinx`](https://www.sphinx-doc.org/en/master/) style docstrings in Python code.

### Pull request

Contributions via pull requests (PR) are welcome.
Push your changes to your fork and submit a pull request.

## Misc

A more comprehensive contribution guide will be added soon.

## Future directions & ToDo's

- [ ] add transfer learning protocols
- [ ] model interpreter should check the analyzer type for compatibility with the model
- [ ] add more model architectures
  - [ ] also add their pretrained versions
- [x] add proper docs
  - [ ] add examples, tutorials, and/or notebooks
- [ ] see ToDo's in code & open issues
