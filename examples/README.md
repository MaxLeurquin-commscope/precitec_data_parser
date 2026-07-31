# Examples

This directory contains example scripts demonstrating how to use `precitec-data-parser`.

## demo.py

A comprehensive demo showing all main features of the library:

- Loading and parsing `.csv` and `.bcrf` files
- Plotting 2D height maps and 3D surfaces
- Extracting horizontal, vertical, and oblique profiles
- Filtering profiles with Gaussian and Hampel filters
- Analyzing surface roughness parameters

### Usage

Update the `my_data` variable to point to your Precitec measurement file:

```python
my_data = "path/to/your/measurement.csv"  # or .bcrf
```

Then run:

```bash
python examples/demo.py
```

### Requirements

- The example requires actual Precitec measurement data (`.csv` or `.bcrf` files)
- Install the package with visualization dependencies:
  ```bash
  pip install precitec-data-parser
  ```

## Testing

For unit tests that don't require data files, see:

```bash
pytest tests/
```

## Contributing

If you have interesting use cases or examples to share, please consider contributing them!
