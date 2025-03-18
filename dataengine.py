import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import os

class DataAnalyzer:
    def __init__(self, data_path=None):
        """Initialize the DataAnalyzer with an optional data path."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.is_classification = None
        
        if data_path:
            self.load_data(data_path)
            
    def load_data(self, path):
        """Load data from CSV, Excel, or other supported formats."""
        print(f"Loading data from {path}")
        if path.endswith('.csv'):
            self.data = pd.read_csv(path)
        elif path.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(path)
        elif path.endswith('.json'):
            self.data = pd.read_json(path)
        else:
            raise ValueError("Unsupported file format. Supported formats: CSV, Excel, JSON")
        
        print(f"Data loaded. Shape: {self.data.shape}")
        return self
        
    def explore_data(self):
        """Perform basic exploratory data analysis."""
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return None
            
        analysis = {}
        
        # Basic info
        print("\n=== Basic Information ===")
        print(f"Number of rows: {self.data.shape[0]}")
        print(f"Number of columns: {self.data.shape[1]}")
        print("\n=== Data Types ===")
        print(self.data.dtypes)
        
        # Missing values
        print("\n=== Missing Values ===")
        missing = self.data.isnull().sum()
        print(missing[missing > 0])
        
        # Statistical summary
        print("\n=== Statistical Summary ===")
        print(self.data.describe())
        
        # For categorical columns
        cat_cols = self.data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print("\n=== Categorical Columns ===")
            for col in cat_cols:
                print(f"\nColumn: {col}")
                print(self.data[col].value_counts())
        
        return self
    
    def clean_data(self, drop_na=False, fill_strategy=None, drop_columns=None):
        """Clean the data by handling missing values and removing specified columns."""
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return None
            
        # Drop specified columns
        if drop_columns:
            self.data = self.data.drop(columns=drop_columns)
            print(f"Dropped columns: {drop_columns}")
            
        # Handle missing values
        if drop_na:
            rows_before = len(self.data)
            self.data = self.data.dropna()
            print(f"Dropped {rows_before - len(self.data)} rows with missing values.")
        elif fill_strategy:
            for col in self.data.columns:
                if self.data[col].dtype.kind in 'ifc':  # Integer, float, complex
                    if fill_strategy == 'mean':
                        self.data[col] = self.data[col].fillna(self.data[col].mean())
                    elif fill_strategy == 'median':
                        self.data[col] = self.data[col].fillna(self.data[col].median())
                    elif fill_strategy == 'zero':
                        self.data[col] = self.data[col].fillna(0)
                else:
                    # For categorical columns, fill with mode
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0] if not self.data[col].mode().empty else "Unknown")
            print(f"Filled missing values using {fill_strategy} strategy.")
            
        return self
    
    def visualize_distributions(self, columns=None):
        """Visualize the distribution of specified numeric columns."""
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return None
            
        if columns is None:
            columns = self.data.select_dtypes(include=['number']).columns[:5]  # First 5 numeric columns by default
            
        num_cols = len(columns)
        if num_cols == 0:
            print("No numeric columns to visualize.")
            return self
            
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 4 * num_cols))
        
        # Handle single column case
        if num_cols == 1:
            axes = axes.reshape(1, 2)
            
        for i, col in enumerate(columns):
            if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col]):
                # Histogram
                sns.histplot(self.data[col], kde=True, ax=axes[i, 0])
                axes[i, 0].set_title(f'Distribution of {col}')
                
                # Box plot
                sns.boxplot(y=self.data[col], ax=axes[i, 1])
                axes[i, 1].set_title(f'Box Plot of {col}')
            else:
                print(f"Column '{col}' is not numeric or doesn't exist.")
                
        plt.tight_layout()
        plt.show()
        return self
    
    def correlation_analysis(self):
        """Generate a correlation matrix and heatmap for numeric columns."""
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return None
            
        numeric_data = self.data.select_dtypes(include=['number'])
        if numeric_data.shape[1] < 2:
            print("Need at least 2 numeric columns for correlation analysis.")
            return self
            
        corr_matrix = numeric_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return self
    
    def prepare_ml(self, target_column, features=None, test_size=0.2, random_state=42):
        """Prepare data for machine learning by splitting into train and test sets."""
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return None
            
        if target_column not in self.data.columns:
            print(f"Target column '{target_column}' not found in data.")
            return None
            
        # Determine features
        if features is None:
            features = [col for col in self.data.columns if col != target_column]
        
        # Check for categorical features and convert them
        X = pd.get_dummies(self.data[features], drop_first=True)
        y = self.data[target_column]
        
        # Determine if classification or regression
        unique_values = len(y.unique())
        self.is_classification = unique_values < 10  # Heuristic: if fewer than 10 unique values, assume classification
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data prepared for {'classification' if self.is_classification else 'regression'}")
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        
        return self
    
    def train_model(self, model_type='auto', **kwargs):
        """Train a machine learning model."""
        if self.X_train is None or self.y_train is None:
            print("Data not prepared. Use prepare_ml() first.")
            return None
            
        # Determine model type if 'auto'
        if model_type == 'auto':
            if self.is_classification:
                model_type = 'random_forest'
            else:
                model_type = 'linear_regression'
        
        # Initialize the model
        if model_type == 'linear_regression':
            self.model = LinearRegression(**kwargs)
            print("Using Linear Regression model")
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(**kwargs)
            print("Using Logistic Regression model")
        elif model_type == 'random_forest':
            if self.is_classification:
                self.model = RandomForestClassifier(**kwargs)
                print("Using Random Forest Classifier")
            else:
                self.model = RandomForestRegressor(**kwargs)
                print("Using Random Forest Regressor")
        else:
            raise ValueError("Unsupported model type. Use 'linear_regression', 'logistic_regression', or 'random_forest'")
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")
        
        return self
    
    def evaluate_model(self):
        """Evaluate the model performance."""
        if self.model is None:
            print("No model trained. Use train_model() first.")
            return None
            
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Evaluate based on problem type
        if self.is_classification:
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(report)
        else:
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Plot actual vs predicted
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.7)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            plt.tight_layout()
            plt.show()
            
        return self
    
    def feature_importance(self):
        """Visualize feature importance for tree-based models."""
        if self.model is None:
            print("No model trained. Use train_model() first.")
            return None
            
        # Check if model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.X_train.columns
            
            # Create DataFrame for easier sorting
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
        else:
            print("This model doesn't support feature importance visualization.")
            
        return self
    
    def pca_analysis(self, n_components=2):
        """Perform PCA and visualize the results."""
        if self.X_train is None:
            print("Data not prepared. Use prepare_ml() first.")
            return None
            
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")
        
        # Visualize
        plt.figure(figsize=(10, 8))
        
        if n_components == 2:
            if self.is_classification:
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train, cmap='viridis', alpha=0.8)
                plt.colorbar(label='Target')
            else:
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train, cmap='coolwarm', alpha=0.8)
                plt.colorbar(label='Target Value')
                
            plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.4f})')
            plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.4f})')
            plt.title('PCA Analysis')
            plt.tight_layout()
            plt.show()
        elif n_components == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                               c=self.y_train, cmap='viridis' if self.is_classification else 'coolwarm', alpha=0.8)
            
            plt.colorbar(scatter, label='Target' if self.is_classification else 'Target Value')
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.4f})')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.4f})')
            ax.set_zlabel(f'PC3 ({explained_variance[2]:.4f})')
            plt.title('3D PCA Analysis')
            plt.tight_layout()
            plt.show()
            
        return self
    
    def clustering_analysis(self, n_clusters=3):
        """Perform K-means clustering analysis."""
        if self.X_train is None:
            print("Data not prepared. Use prepare_ml() first.")
            return None
            
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Visualize clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.8)
        
        # Add cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, alpha=0.8)
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('K-means Clustering Analysis')
        plt.tight_layout()
        plt.show()
        
        return self
    
    def save_model(self, filename):
        """Save the trained model to disk."""
        if self.model is None:
            print("No model trained. Use train_model() first.")
            return None
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
        
        return self
    
    def load_model(self, filename):
        """Load a previously saved model."""
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return self
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    # Generate some sample data for demonstration
    print("Generating sample data...")
    np.random.seed(42)
    
    # Create synthetic dataset
    n_samples = 1000
    
    # Features
    age = np.random.normal(35, 10, n_samples)
    experience = age - 18 + np.random.normal(0, 2, n_samples)
    experience = np.maximum(0, experience)  # No negative experience
    education = np.random.choice([10, 12, 16, 18, 20], n_samples)
    skills = np.random.normal(7, 2, n_samples)
    
    # Target (salary) with some noise
    salary = 30000 + 2000 * experience + 3000 * education + 5000 * skills + np.random.normal(0, 10000, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Experience': experience,
        'Education': education,
        'Skills': skills,
        'Salary': salary
    })
    
    # Save to CSV
    data.to_csv('salary_data.csv', index=False)
    print("Sample data saved to 'salary_data.csv'")
    
    # Demonstrate full analysis pipeline
    analyzer.load_data('salary_data.csv')
    analyzer.explore_data()
    analyzer.clean_data(fill_strategy='mean')
    analyzer.visualize_distributions()
    analyzer.correlation_analysis()
    analyzer.prepare_ml(target_column='Salary')
    analyzer.train_model(model_type='random_forest', n_estimators=100)
    analyzer.evaluate_model()
    analyzer.feature_importance()
    analyzer.pca_analysis()
    analyzer.clustering_analysis(n_clusters=3)
    analyzer.save_model('salary_prediction_model.pkl')
    
    print("\nAnalysis complete!")
