# Documentation for the project

This is the documentation folder for the project. It uses sphinx to generate the documentation.

## Building the documentation locally

To build the documentation locally, you need to have sphinx installed. You can install it using pip:

```bash
pip install -r requirements.txt
```

Then, you can build the documentation by running the following command:

```bash
sphinx-build -b html . public
```

You can then host locally the documentation by running the following command:

```bash
python -m http.server -d public
```

## Docstrings style

We are using the Google docstrings style (via the `sphinx.ext.napoleon` extension). Here is an example of a function with a docstring:
```python
def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """
    return True
```
