#!/bin/bash

lr1=(0.0005  0.0005  0.0005  0.0010 0.0005 0.0003)
lr2=(0.0005  0.0010  0.0010  0.0005 0.0005 0.0003)
lr3=(0.0010  0.0010  0.0005  0.0005 0.0005 0.0003)
# lr1=(0.001)
# lr2=(0.001)
# lr3=(0.001)
len=${#lr1[*]}


for i in `seq 0 $((len-1))`; do 
    V_LR1=${lr1[$i]}
    V_LR2=${lr2[$i]}
    V_LR3=${lr3[$i]}
    mkdir -p lr_${V_LR1}_${V_LR2}_${V_LR3}
    cd lr_${V_LR1}_${V_LR2}_${V_LR3}
    cp ../* .
    cat <<EOF > lr_${V_LR1}_${V_LR2}_${V_LR3}.sh
#!/bin/bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=12:00:00

module purge
source ~/.bashrc
conda activate imolcry
python3 rdf.py -lr ${V_LR1} ${V_LR2} ${V_LR3}
EOF
    rsub -i lr_${V_LR1}_${V_LR2}_${V_LR3}.sh
    cd ..
done

