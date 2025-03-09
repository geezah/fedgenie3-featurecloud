# Federated GENIE3 for FeatureCloud

This is an implementation of federated GENIE3 (GEne Network Inference with Ensemble of trees) for the FeatureCloud platform. It enables privacy-preserving gene regulatory network (GRN) inference across multiple data sites without sharing raw data.

## Description

Federated GENIE3 extends the original GENIE3 approach to a distributed setting. Like GENIE3, it treats network inference as a feature selection problem: each site predicts gene expression levels using other genes (especially transcription factors) as features. However, instead of computing importance scores centrally, each site calculates them locally using their private gene expression data. These local importance scores are then securely aggregated to estimate regulatory link strengths across the entire network. This federated approach enables multiple institutions to collaboratively infer gene regulatory networks while keeping their sensitive genomic data private and local, ultimately resulting in a more accurate and robust network inference.

## Usage on FeatureCloud

### Features

- **Privacy-Preserving**: Raw gene expression data never leaves local sites
- **Multiple Aggregation Strategies**:
  - Sample-size weighting (Weighting based on number of samples)
  - Invariance weighting (Weighting based on number of samples + invariance of the feature importances across sites)
- **Multiple Regressor Support**: Supports various tree-based ensemble methods:
  - Random Forest
  - Extra Trees
  - Gradient Boosting

### Input

- Gene expression data: A tab-separated file with genes as columns and samples as rows.
- Transcription factor list: A tab-separated file with one column containing transcription factor names.
- Coordinator Configuration: A yaml file specifying global parameters, for instance:
  
  ```yaml
  aggregation:
    name: "invariance-weighting"  # Options: sample-size-weighting, invariance-weighting. # See src/aggregation.py
    params: 
      decay_exponent: 2 
  regressor:
    name: "ExtraTreesRegressor"  # Options: See https://github.com/geezah/genie3/blob/main/genie3/regressor/__init__.py
  ```

- Client Configuration: A yaml file specifying names of the tab-separated files containing gene expression levels and transcription factors respectively, for instance:
  
  ```yaml
  data:
    gene_expressions_path: name_of_gene_expression_file.tsv
    transcription_factors_path: name_of_transcription_factors_file.tsv # Optional, but recommended. If not provided, the application will consider all genes as transcription factors.
  ```

For detailed information on the configuration schema, see [the configuration schema of this application](src/config.py) and [the configuration schema of the core package](https://github.com/geezah/genie3/blob/main/genie3/config.py).  

### Output

- Predicted network: A tab-separated file with three columns for transcription factors, target genes, and importance scores.

### Data Format

The expected format of the data files. The header rows are expected to be present in the respective files.

#### Gene Expression Data

A tab-separated file with genes as columns, samples as rows, and gene expression values as entries:

```csv
        Gene1   Gene2   Gene3   ...  # Header row
Sample1 0.5     1.2     0.8     ...
Sample2 0.7     0.9     1.1     ...
...
```

#### Transcription Factor List

A tab-separated file with one column containing transcription factor names. The transcription factors are expected to be present in the columns of the gene expression data.

```csv
transcription_factor  # Header row
Gene1
Gene2
...
```

## Development

### Requirements

- Python 3.12
- [Docker](https://docs.docker.com/get-docker/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python environment manager)
- [FeatureCloud](https://featurecloud.ai/) platform

### Setting up the environment

1. Install the Python environment manager `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. In the root folder of this repository, run:
   ```bash
   uv sync
   ```

3. Build the FeatureCloud container:
   
   ```bash
   featurecloud app build . fedgenie3 latest True
   ```

4. Launch the controller and specify the directory containing the configuration and data files, e.g. `./controller_data`:
   
   ```bash
   featurecloud controller start --data-dir=./controller_data
   ```

5. Assume the structure of the `controller_data` directory is as follows:

   ```
   controller_data/
   ┣ 📂clients
   ┃ ┣ 📂client1
   ┃ ┃ ┣ client.yaml
   ┃ ┃ ┣ gene_expressions_data.tsv
   ┃ ┃ ┗ transcription_factors.tsv
   ┃ ┗ 📂client2
   ┃   ┣ client.yaml
   ┃   ┣ gene_expressions_data.tsv
   ┃   ┗ transcription_factors.tsv
   ┗ 📂generic
     ┗ 📜server.yaml 
   ```

   In this case, the files in the `generic` directory and only the client-specific files in the respective client directories  are mounted at `/mnt/input/` for each client. Here, the path configurations in the `client.yaml` files are only the names of the files, for instance resulting in `/mnt/input/name_of_gene_expression_data.tsv` in the container. 
   
   Then, to run the application in the testbed, you can use the following command to launch 2 clients:

   ```bash
   featurecloud test start --app-image fedgenie3 \
     --client-dirs="controller_data/clients/client1,controller_data/clients/client2" \
     --generic-dir="controller_data/generic"
   ```
  
  Launching the tests on the testbed web interface is also possible and the recommended approach, since it is more convenient. See [here](https://featurecloud.ai/development/test). Note that the controller must be running in order to start testing.

### Asynchronous Simulation

For experimenting locally on simulated partitions of a single dataset without the full FeatureCloud infrastructure, you can use the provided asynchronous simulation script based on `asyncio`:

```bash
uv run python3 async_fed_sim.py client.yaml server.yaml 2
```

This simulates federated learning with 2 sites using the specified configurations by partitioning the dataset into 2 parts. 

Multiple simulation strategies can be specified in the `server.yaml` file:

- `random-even`: The dataset is randomly and equally-sized partitioned into the specified number of parts.
- `tf-centric`: The dataset is partitioned based on the results of an agglomerative clustering approach, with the number of clusters being equal to the specified number of sites.

Multiple aggregation strategies can be specified in the `server.yaml` file, allowing for faster experimentation and resulting in distinct runs with distinct outputs. For instance:

```yaml
aggregation:
  - name: "sample-size-weighting"
    params: 
  - name: "invariance-weighting"
    params: 
      decay_exponent: 2
simulation:
  - name: "random-even"  # Options: random-even, tf-centric. See src/simulation.py for more details.
```

The `client.yaml` file specifies the path to the gene expression file, the transcription factor file, and the reference network file.

```yaml
data:
  gene_expressions_path: /path/to/gene_expressions_data.tsv
  transcription_factors_path: /path/to/transcription_factors.tsv
  reference_network_path: /path/to/reference_network.tsv
```

The output will look like the following:

```
📦2025-03-08_10-11-05
 ┣ 📜coordinator_config.yaml
 ┣ 📜global_network_invariance-weighting.csv
 ┣ 📜global_network_sample-size-weighting.csv
 ┣ 📜local_network_1.csv
 ┣ 📜local_network_2.csv
 ┣ 📜metrics.csv
 ┣ 📜participant_config.yaml
 ┗ 📜reference_network.csv
```


### Data Format

The expected format of the data files. The header rows are expected to be present in the respective files.

#### Gene Expression Data

See above

#### Transcription Factor List

See above

#### Reference Network

The additional reference network file that will be evaluated against is a tab-separated file with columns for transcription factors, target genes, and binary labels, indicating the presence of an edge between the transcription factor and the target gene:

```csv
transcription_factor  target_gene  label
Gene1                 Gene2        1
Gene1                 Gene3        0
...
```

## Citation

If you use this implementation in your research, please cite the original GENIE3 paper:

```bibtex
@article{huynh-thuInferringRegulatoryNetworks2010,
  title = {Inferring {{Regulatory Networks}} from {{Expression Data Using Tree-Based Methods}}},
  author = {{Huynh-Thu}, V{\^a}n Anh and Irrthum, Alexandre and Wehenkel, Louis and Geurts, Pierre},
  editor = {Isalan, Mark},
  year = {2010},
  month = sep,
  journal = {PLoS ONE},
  volume = {5},
  number = {9},
  pages = {e12776},
  issn = {1932-6203},
  doi = {10.1371/journal.pone.0012776},
  urldate = {2024-09-23},
  langid = {english},
  keywords = {Gene Regulatory Networks}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
