import os, psutil, tracemalloc
from functools import partial
from imolcraft.trainer import ThermodynamicTrainer
from imolcraft.trainer.loss import loss_thermodynamicperturbation

sampling_params = {'init_structure': 'merged_supercell_bonds.pdb',
                   'ensemble': 'nvt',
                   'dt_fs': 1.0,
                   'rcut_nm': 1.2,
                   'temperature_K': 233.15,
                   'pressure_bar': 1.0,
                   'anneal_steps': 1,
                   'anneal_Tmax': 400.0,
                   'anneal_totalsteps': 0,
                   'relax_steps': 2,
                   'prod_steps': 2,
                   'nstxout': 1,
                   'neff': 2,
                   'anneal_totaltime': 0}

target_params = {'density_gcm3': {'weight': 1.0, 'gt': 0.4},
                 'La_A': {'weight': 1.0, 'gt': 41},
                 'Lb_A': {'weight': 1.0, 'gt': 41},
                 'Lc_A': {'weight': 1.0, 'gt': 41}}

lossfn = partial(loss_thermodynamicperturbation, losstype_distribfn="wrightfactor")

trainer = ThermodynamicTrainer(
    ffxml_list=['tests/data/vsite_average2.xml'],
    nums_ffxml=[1],
    pdbfile='supercell_bonds.pdb',
    loss_fn=lossfn,
    sampling_params=sampling_params,
    target_params=target_params,
    opt_fftypes=["NonbondedForce/charge", 'VirtualSite/weight'],
    label='debug_tp',
    lr=0.0001
)

print('setup...')
trainer.setup()

proc = psutil.Process(os.getpid())
tracemalloc.start()

for step in range(3):
    trainer.before_step()
    trainer.training_step()
    trainer.after_step()
    mem = proc.memory_info().rss / (1024*1024)
    current, peak = tracemalloc.get_traced_memory()
    print(f'Step {step}: RSS={mem:.2f} MB, tracemalloc_current={current/1024/1024:.2f} MB, peak={peak/1024/1024:.2f} MB')

print('done')
