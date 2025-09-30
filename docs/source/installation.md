# Installation
## Procedure
### Stable version (Mac and Linux)
```
git clone git@github.com:srak-uf/iMolCRAFT.git
cd iMolCRAFT
conda env create -f env.yml
conda activate imc
pip install .
```

### Developer version (Mac and Linux)
1. Create conda environment
    ```
    git clone git@github.com:srak-uf/iMolCRAFT.git
    cd iMolCRAFT
    conda env create -f env_dev.yml
    conda activate imc_dev
    ```

1. Installation of DMFF
    ```
    git clone https://github.com/srak-uf/DMFF
    cd DMFF
    pip install -e . 
    cd ..
    ```

1. Installation of openff-recharge
    ```
    git clone https://github.com/srak-uf/openff-recharge
    cd openff-recharge
    pip install -e .
    cd ..
    ```

1. Installation of iMolCRAFT
    ```
    pip install -e .
    ```

### Make document
```
conda install -c conda-forge sphinx myst-parser sphinx_rtd_theme
cd docs
make html
```

### Testing
```
conda install -c conda-forge pytest
cd tests
pytest *
```

## Troubleshooting/issues
Contact us at [issues](https://github.com/srak-uf/iMolCRAFT/issues) or by [email](<mailto:sasaki.r.1788@m.isct.ac.jp>) when you have trouble.
