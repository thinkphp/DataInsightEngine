# DataInsightEngine

A comprehensive Python application for data analysis, visualization, and machine learning using pandas, matplotlib, seaborn, and scikit-learn.

## Features

- **Data Loading & Exploration**
  - Support for CSV, Excel, and JSON formats
  - Automatic data profiling and summary statistics
  - Missing value detection and handling

- **Data Visualization**
  - Distribution analysis with histograms and box plots
  - Correlation matrix heatmaps
  - Feature importance charts
  - Principal Component Analysis (PCA) visualization
  - Cluster visualization

- **Machine Learning**
  - Automatic detection of classification vs. regression problems
  - Support for multiple algorithms:
    - Linear Regression
    - Logistic Regression
    - Random Forest (Classification & Regression)
  - Model evaluation with appropriate metrics
  - Feature importance analysis
  - Dimensionality reduction with PCA
  - Clustering analysis with K-means
  - Model saving and loading functionality

## Installation

```bash
# Clone the repository
git clone https://github.com/thinkphp/DataInsightEngine.git
cd DataInsightEngine

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage

### Basic Usage

```python
from data_analyzer import DataAnalyzer

# Initialize the analyzer
analyzer = DataAnalyzer()

# Load data
analyzer.load_data('your_data.csv')

# Explore and clean data
analyzer.explore_data()
analyzer.clean_data(fill_strategy='mean')

# Visualize data
analyzer.visualize_distributions()
analyzer.correlation_analysis()

# Prepare data for machine learning
analyzer.prepare_ml(target_column='your_target_column')

# Train and evaluate model
analyzer.train_model(model_type='random_forest')
analyzer.evaluate_model()
analyzer.feature_importance()

# Save model for future use
analyzer.save_model('your_model.pkl')
```

### Advanced Usage

```python
# Load existing model
analyzer.load_model('your_model.pkl')

# Advanced visualization
analyzer.pca_analysis(n_components=3)
analyzer.clustering_analysis(n_clusters=5)

# Customize model parameters
analyzer.train_model(
    model_type='random_forest',
    n_estimators=200,
    max_depth=10,
    random_state=42
)
```

## Example

The package includes a sample implementation that:
1. Generates synthetic salary prediction data
2. Performs exploratory data analysis
3. Trains a Random Forest model
4. Evaluates and visualizes the results

To run the example:

```python
python data_analyzer.py
```

## Class Reference

### DataAnalyzer

```python
analyzer = DataAnalyzer(data_path=None)
```

#### Methods

- `load_data(path)`: Load data from file
- `explore_data()`: Perform exploratory data analysis
- `clean_data(drop_na=False, fill_strategy=None, drop_columns=None)`: Clean the dataset
- `visualize_distributions(columns=None)`: Create distribution plots
- `correlation_analysis()`: Generate correlation heatmap
- `prepare_ml(target_column, features=None, test_size=0.2, random_state=42)`: Prepare data for ML
- `train_model(model_type='auto', **kwargs)`: Train a machine learning model
- `evaluate_model()`: Evaluate model performance
- `feature_importance()`: Visualize feature importance
- `pca_analysis(n_components=2)`: Perform PCA analysis
- `clustering_analysis(n_clusters=3)`: Perform K-means clustering
- `save_model(filename)`: Save model to disk
- `load_model(filename)`: Load model from disk

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

- [thinkphp](https://github.com/thinkphp)
