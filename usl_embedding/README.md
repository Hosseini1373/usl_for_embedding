
# Unsupervised Learning for Embedding (USL)

This project implements the USL and USL-t methods for unsupervised learning with embeddings, allowing the use of embeddings to form clusters and potentially improve learning outcomes in unsupervised settings.

## Installation

Clone the repository and install the required packages:

```bash
git clone [URL_to_this_repo]
cd [repo_name]
pip install -r requirements.txt
```

## Usage

The `usl_for_embedding` function is designed to process embedding arrays for clustering. It can be used directly by importing from the `selective_labeling.py`:

```python
from selective_labeling import usl_for_embedding

# Example usage:
embeddings = # your embedding data here
method = 'usl'  # or 'usl-t'
n_clusters = 5  # number of clusters to form

selected_indices = usl_for_embedding(embeddings, method=method, n_clusters=n_clusters)
print("Selected indices for clustering:", selected_indices)
```

### Parameters
- **embeddings (array-like)**: Input data, embeddings to be clustered.
- **method (str)**: The clustering method to use ('usl' or 'usl-t').
- **n_clusters (int)**: Number of clusters to form.
- Additional parameters include learning rate, batch size, initialization runs, regularization term, and more described in `main.py`.

## Project Structure

```
project_root/
│
├── methods/                # Module for clustering algorithms
│   ├── usl.py              # Contains the density regulation based USL
│   └── usl_t.py            # Contains the transformer based USL-t
│
├── models/                 # Models used in USL-t
│   └── ssl_t_models/
│       └── clustering_model.py  # Clustering model for USL-t
│
└── main.py                 # Main script to use USL methods
```

## Main Files and Their Roles

- **main.py**: Contains the primary functions used to interface with the USL methods and demonstrates an example usage.
- **usl.py** and **usl_t.py**: Implementation of the USL and USL-t methods, respectively.
- **clustering_model.py**: Defines the neural network models used in the USL-t method.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your features or fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

Replace `[URL_to_this_repo]` and `[repo_name]` with your actual repository URL and name. This README provides a clear overview of how to set up and use the project, and details the files and their functions within the project structure.