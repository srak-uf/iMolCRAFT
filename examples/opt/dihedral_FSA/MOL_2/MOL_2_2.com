%mem=32GB
%nprocshared=32
#P wb97xd/6-311++g(d,p) ! ASE formatted method and basis
opt(maxcycle=256)

Gaussian input prepared by ASE

1 1
Li                6.0470000000        6.0470000000        5.0269000000


