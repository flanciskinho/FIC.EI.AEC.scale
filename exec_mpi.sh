PROG=mpi_omp_sse
NPROCS=3
mpirun -np ${NPROCS} ./${PROG}.out 80 80 2 2
