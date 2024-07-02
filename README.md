# Taylor Diagram

A Taylor Diagram is a graphical representation used to evaluate how well models reproduce the observed data. It displays three statistical measures: correlation coefficient, root mean square deviation (RMSD), and the standard deviation of both observed and model data. Here’s a basic guide on how to create and interpret a Taylor Diagram.
This repository contains the code for generating Taylor Diagrams to visualize the performance of various models based on multiple evaluation metrics. The Taylor Diagrams are created from a CSV file containing the observed values and model predictions.

## Features

- Generates Taylor Diagrams for multiple models from a CSV file.
- Customizable markers and marker sizes.
- Interactive plots with standard deviation and correlation contours.

## Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

- `numpy`
- `matplotlib`
- `pandas`

You can install these packages using `pip`:

```bash
pip install numpy matplotlib pandas
```

### Repository Structure

- `TaylorDiagram.py`: Contains the `TaylorDiagram` class and script to generate Taylor Diagrams.
- `data.csv`: Sample CSV file with observed values and model predictions.

### Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/EbrahimAlwajih/Taylor-Diagram.git
   cd Taylor-Diagram
   ```

2. **Prepare your CSV file:**

   Ensure your CSV file (`data.csv`) has the following structure:
   
   ```csv
   model,observed,Model 1,Model 2,Model 3,Model 4,Model 5
   value1,observed_value,Model 1_value,Model 2_value,Model 3_value,Model 4_value,Model 5_value
   value2,observed_value,Model 1_value,Model 2_value,Model 3_value,Model 4_value,Model 5_value
   ...
   ```

3. **Run the script:**

   Execute the script to generate the Taylor Diagrams:

   ```bash
   python TaylorDiagram.py
   ```

   The Taylor Diagrams will be displayed in a new window.

### Customization

- **Markers:**
  You can customize the markers for the radar charts by modifying the `markers` parameter in the `TaylorDiagram` class.

  ```python
  markers = ['*', 's', '^', 'D', 'x']
  ```

- **Marker Size:**
  You can adjust the size of the markers by changing the `marker_size` parameter.

  ```python
  marker_size = 10
  ```

## License

This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

If you have any questions, feel free to reach out to the repository owner.

