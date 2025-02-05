#!/bin/bash

# gros=(1_epoch-215.gro 2_epoch-415.gro)
gros=(`ls *gro`)
len_g=${#gros[*]}

for i in `seq 0 $((len_g-1))`; do
    basename=${gros[$i]%*.gro}
    echo $basename
    # gmx grompp -f tri.mdp -p ${basename}.top -c ${basename}.gro -o ${basename}_tri.tpr
    # gmx mdrun -deffnm ${basename}_tri -ntmpi 4 -ntomp 2 -v
    # python3 mda_rdf.py -f ${basename}_tri.trr -s merged_supercell_bonds.pdb -e1 Li -e2 N
    # python3 mda_rdf.py -f ${basename}_tri.trr -s merged_supercell_bonds.pdb -e1 Li -e2 O
    cat <<EOF > j${basename}_md.sh
#!/bin/sh
#$ -cwd
#$ -l cpu_80=1
#$ -l h_rt=23:10:00

source ~/.bashrc
module purge
module load intel/2024.0.2 intel-mpi/2021.11

gmx_mpi grompp -f min.mdp -p ${basename}.top -c ${basename}.gro -o ${basename}_min.tpr
mpirun -np 24 gmx_mpi mdrun -deffnm ${basename}_min -ntomp 3 -v

gmx_mpi grompp -f nvt.mdp -p ${basename}.top -c ${basename}_min.gro -o ${basename}_nvt.tpr
mpirun -np 24 gmx_mpi mdrun -deffnm ${basename}_nvt -ntomp 3 -v

gmx_mpi grompp -f anisonpt_xyz.mdp -p ${basename}.top -c ${basename}_nvt.gro -o ${basename}_aniso.tpr
mpirun -np 24 gmx_mpi mdrun -deffnm ${basename}_aniso -ntomp 3 -v

gmx_mpi grompp -f trinpt_xyz_xy_yz_zx.mdp -p ${basename}.top -c ${basename}_aniso.gro -o ${basename}_tri.tpr
mpirun -np 24 gmx_mpi mdrun -deffnm ${basename}_tri -ntomp 3 -v

conda activate mi
python3 mda_rdf.py -f ${basename}_tri.trr -s merged_supercell_bonds.pdb -e1 Li -e2 N
python3 mda_rdf.py -f ${basename}_tri.trr -s merged_supercell_bonds.pdb -e1 Li -e2 O
EOF
    rsub -i j${basename}_md.sh
done
