# Installation
## Procedure
### 1. Clone iMolCRAFT from GitHub and create conda virtual environment
```
git clone git@github.com:srak-uf/imolcryff.git
cd imolcryff

conda env create -f imolcry_dev.yml # this script can be used in linux
conda activate imc_dev
```

### 2. Clone DMFF from GitHub and pip install
```
git clone https://github.com/srak-uf/DMFF
cd DMFF
pip install -e . 
```

### 3. Install iMolCRAFT
```
cd ..
pip install -e . 
```

## Troubleshooting/issues
Contact us at issues or by email when you have trouble.
