# Installation
```
conda env create -f imolcry.yml # this script can be used in linux
# Installation of DMFF
git clone https://github.com/srak-uf/DMFF
cd DMFF
pip install . --user
cd ..

# Installation of iMolCryFF
git clone git@github.com:srak-uf/imolcryff.git
cd imolcryff
pip install . --user
```
Additionally, you can optimize force field parameters by using [DMFF](https://github.com/srak-uf/DMFF). This DMFF repository is the modified version.    
The sample codes for the optimization are located in [the example/opt](https://github.com/srak-uf/imolcryff/tree/main/examples/opt) directory.
