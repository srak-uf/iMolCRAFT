# Installation of stable version
```
git clone git@github.com:srak-uf/iMolCRAFT.git
cd iMolCRAFT
conda env create -f env.yml # this script can be used in linux and mac
conda activate imc
pip install .
```

# Installation of developer version
```
git clone git@github.com:srak-uf/iMolCRAFT.git
cd iMolCRAFT
conda env create -f env_dev.yml # this script can be used in linux and mac
conda activate imc_dev
git clone https://github.com/srak-uf/DMFF
cd DMFF
pip install -e . 
cd ..

git clone https://github.com/srak-uf/openff-recharge
cd openff-recharge
pip install -e .
cd ..

# Installation of iMolCRAFT
pip install -e .

# make document (Optional)
conda install -c conda-forge sphinx myst-parser sphinx_rtd_theme myst-nb
cd docs
make html

# test
conda install -c conda-forge pytest
```
Additionally, you can optimize force field parameters by using [DMFF](https://github.com/srak-uf/DMFF). This DMFF repository is the modified version.    
The sample codes for the optimization are located in [the example/opt](https://github.com/srak-uf/imolcryff/tree/main/examples/opt) directory.
