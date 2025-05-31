# Simplified PyTorch Implementation of Deep SVDD

This [PyTorch](https://pytorch.org/) implementation of the *Deep SVDD* (Deep Support Vector Data Description) method for Unsupervised Anomaly Detection is a simplified version of the original code available at [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch).

## Overview

Deep SVDD is an unsupervised anomaly detection algorithm that maps data into a hypersphere of minimum volume in a latent space. The algorithm aims to learn a neural network transformation that maps normal data points close to the center of this hypersphere while anomalies are expected to be mapped outside or far from the center.

The project includes:
- Implementation of both One-Class and Soft-Boundary Deep SVDD variants
- Autoencoder pretraining for better initialization
- Visualization tools for both latent space and data space
- Evaluation metrics for anomaly detection performance

## Algorithm Description

The Deep SVDD algorithm works in two main phases:
1. **Pretraining Phase**: An autoencoder is trained to learn a meaningful representation of the data.
2. **Training Phase**: The encoder part of the autoencoder is used to initialize the Deep SVDD network, which is then trained to minimize the volume of a hypersphere that encloses the normal data points.

The objective function for One-Class Deep SVDD is:
- Minimize the mean squared distance of the mapped data points to the center of the hypersphere.

The objective function for Soft-Boundary Deep SVDD is:
- Minimize the hypersphere radius while allowing a small fraction of data points to fall outside the hypersphere.

## Requirements
- Python 3.12
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Scikit-learn >= 1.2.0
- tqdm >= 4.65.0
- SciPy >= 1.10.0

All dependencies are listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rjsaltos1989/Deep-SVDD.git
   cd D-SVDD
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n dsvdd python=3.12
   conda activate dsvdd
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `main.py`: Main script to run the Deep SVDD algorithm
- `nn_models.py`: Neural network model definitions (AutoEncoder and DeepSVDD)
- `nn_train_functions.py`: Functions for training the autoencoder
- `svdd_nn_train_functions.py`: Functions for training the Deep SVDD model
- `svdd_eval_functions.py`: Functions for evaluating the Deep SVDD model
- `plot_functions.py`: Functions for visualizing results

## Usage

### Basic Usage

1. Modify the dataset path in `main.py` to point to your data:
   ```python
   dataset_path = '/path/to/your/data/'
   dataset_file = 'YourDataset.mat'
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

### Customization

You can customize the following parameters in `main.py`:

- `latent_dim`: Dimension of the latent space (default: 2)
- `nu`: Hyperparameter controlling the fraction of outliers (default: 0.03)
- `pretrain_epochs`: Number of epochs for autoencoder pretraining (default: 10)
- `train_epochs`: Number of epochs for Deep SVDD training (default: 100)
- `batch_size`: Batch size for training (default: 32)

### Input Data Format

The code expects data in MATLAB .mat format with:
- 'Data': Matrix where rows are samples and columns are features
- 'y': Vector of labels where anomalies are labeled as 2

## Example Results

When running the code, you'll get:
1. Training loss plots showing the convergence of the model
2. Visualization of the data in the latent space, showing:
   - Normal data points
   - Support vectors (points on the hypersphere boundary)
   - Bounded support vectors (points outside the hypersphere)
   - The center of the hypersphere
3. Performance metrics including AUC-ROC, AUC-PR, F1-Score, and Recall

## References

- Original Paper: Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Müller, E., & Kloft, M. (2018). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR.
- Original Implementation: [lukasruff/Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
