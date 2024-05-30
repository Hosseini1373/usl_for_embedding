
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

# Generate 10 embeddings, each with 10 random floating-point numbers
data = {'embedding': [list(np.random.rand(10)) for _ in range(10)]}

# method is 'usl' or 'usl-t'
# n_clusters is the number of samples that should be selected for labeling
selected_indices = usl_for_embedding(data['embedding'],method='usl',n_clusters=2,
                            learning_rate=0.001,batch_size=64,n_init=10,m_reg=0.9,k=10,lambda_=0.5,
                            epsilon=1e-5,alpha=0.75,num_epochs_cluster=100,num_heads=3)

print(selected_indices)
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
This code is used parts of the paper and codebase from repository: 
https://github.com/TonyLianLong/UnsupervisedSelectiveLabeling

## License

This project is licensed under the MIT License - see the LICENSE file for details.
