# Installation
## Procedure
### Create conda env and installation of iMolCRAFT (Mac and Linux)
```
git clone https://github.com/srak-uf/iMolCRAFT.git
cd iMolCRAFT
conda env create -f env.yml
conda activate imc
pip install .
```

### Make document
```
conda install -c conda-forge sphinx myst-parser sphinx_rtd_theme myst-nb
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
