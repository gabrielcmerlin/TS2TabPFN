# TS2TabPFN

## Description

Time series data are ubiquitous in practical applications, where classification (TSC) and extrinsic regression (TSER) have emerged as essential tasks for obtaining value from temporal sequences. While the literature has seen significant progress through feature-based and deep learning models, existing methods often focus either on the quality of feature extraction or on the intrinsic predictive power of complex architectures applied to raw data. This division creates a gap between the control offered by feature engineering and the automated performance of end-to-end models. This paper proposes TS2TabPFN, a framework that bridges this gap by integrating explicit feature extraction with TabPFN 2.5, a cutting-edge foundation model for tabular data, to leverage its predictive capabilities. Our extensive experimental evaluation demonstrates that TS2TabPFN significantly outperforms state-of-the-art models in TSER tasks with statistical significance, providing a robust and efficient alternative for TSC and surpassing most of the currently best-performing algorithms. These results suggest that combining foundation models with structured features overcomes single-paradigm limitations, establishing a new time series state-of-the-art.

## Structure

```bash
TS2TabPFN/
├── code/
│   ├── results/                  # Scripts for plotting comparisons and tracking training
│   └── utils/                    # Helper functions (e.g., data loading, training)
│   └── extract_*.py              # Scripts for extracting features from time series
│   └── main.py                   # Main script for executing the pipeline
├── tsc/                     
│   ├── outputs/                  # Raw training outputs      
│   └── results/                  # Accuracy results formatted for comparison libraries
├── tser/                     
│   ├── outputs/                  # Raw training outputs
│   └── results/                  # RMSE results formatted for comparison libraries
├── .gitignore 
├── README.md                     # Project documentation (this file)
├── config.yaml                   # Configuration file (hyperparameters, paths, dataset settings)
└── env.yaml                      # Conda environment definition for reproducibility
```

## How to start

### Updating system

```bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential
sudo apt install nvidia-cuda-toolkit
```

### Installing Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Downloading code:
```bash
git clone https://github.com/gabrielcmerlin/TS2TabPFN.git
cd TS2TabPFN
```

### Installing requisites:
```bash
conda env create -f env.yaml
conda activate ecml
pip install git+https://github.com/jose-gilberto/labicompare/
```

### Downloading data

```bash
# Download TSER dataset
gdown 1Bg_KHSv77eMbqDyElPohJ1dIam1CkS_V -O tser_data.zip
mkdir tser/data
mv tser_data.zip tser/data
cd tser/data
unzip tser_data.zip
rm tser_data.zip
cd ../..

# Download TSC dataset
gdown 1p8D-haDDwPKloKMBQCQHgENYHVsjixR4 -O tsc_data.zip
mkdir tsc/data
mv tsc_data.zip tsc/data
cd tsc/data
unzip tsc_data.zip
rm tsc_data.zip
cd ../..
```

### Setting up the TabPFN Model
To download the TabPFN 2.5 weights, you must authenticate with Hugging Face using an access token:

1. Log in to your [Hugging Face account](https://huggingface.co/).
2. Navigate to **Settings > Access Tokens** and create a new token with **Read** permissions.
3. Run the following command in your terminal and paste your token when prompted:
```bash
huggingface-cli login
```

### Important Notes

**Note 1**: If you plan to run experiments over multiple runs on the same datasets, it is highly recommended to extract features beforehand rather than recalculating them during every training cycle. This significantly reduces total computation time. You should run the extraction scripts first to generate local feature files, which the model can then load directly:
```bash
# Change * for {c22, multi, tsfresh}
python3 code/extract_*.py -c config.yaml
```

**Note 2**: Please note that while the `.zip` files provided above are hosted on my Drive for convenience, I do not own this data. The files are simply compilations of existing datasets, packaged together to make running this code easier right out of the box. All credit for the datasets goes to their original creators and curators:
* **TSC Data:** Originally sourced from the [UCR/UEA TSC Archive](https://www.timeseriesclassification.com/dataset.php).
* **TSER Data:** Originally sourced from the [TSML Extended
Archive](https://zenodo.org/records/11236865).

## Run

### Experiments
After updating your settings in the ```config.yaml``` file, you can run the commands below.
```bash
python3 code/main.py -c config.yaml
```

### Prepare outputs
The training code generates raw outputs that require formatting before they can be used with comparison libraries. This conversion is handled by a separate script. Update the input and output paths in the file, then run:
```bash
python3 code/results/transform_results.py -c config.yaml
```

### Result Analysis
Run the Python Notebook named 'results.ipynb' located in 'code/results/'