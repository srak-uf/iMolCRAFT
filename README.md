# Installation of developer version
```
git clone git@github.com:srak-uf/iMolCRAFT.git
cd iMolCRAFT
conda env create -f imolcry_dev.yml # this script can be used in linux
conda activate imc_dev
git clone https://github.com/srak-uf/DMFF
cd DMFF
pip install -e . 
cd ..

git clone https://github.com/srak-uf/openff-recharge
cd openff-recharge
pip install -e .
cd ..
pip uninstall dataclasses -y

# Installation of iMolCryFF
pip install -e .

# make document
cd docs
make html
```
Additionally, you can optimize force field parameters by using [DMFF](https://github.com/srak-uf/DMFF). This DMFF repository is the modified version.    
The sample codes for the optimization are located in [the example/opt](https://github.com/srak-uf/imolcryff/tree/main/examples/opt) directory.
