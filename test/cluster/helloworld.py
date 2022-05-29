#!usr/bin/env python
from mpi4py import MPI
from helloworld_unit import helloworld_unit

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name()

n_combination = 4 # totally 4 configurations for helloworld_unit
paraTable = [
    {'p0': 0, 'p1': 1},
    {'p0': 10, 'p1': 11},
    {'p0': 20, 'p1': 21},
    {'p0': 30, 'p1': 31},
] # all possibale configurations for the function helloworld_unit

for i in range(rank, n_combination, size):
    helloworld_unit(rank, node_name, paraTable[i]['p0'], paraTable[i]['p1'])
