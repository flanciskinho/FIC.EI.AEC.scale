#/!bin/bash
CC_MPI=mpicc
CC=gcc
FLAGS="-msse3 -fopenmp -lm"
PROG1=original
PROG2=mpi_omp
PROG3=mpi_omp_sse
${CC} $FLAGS -o ${PROG1}.out ${PROG1}.c
${CC_MPI} $FLAGS -o ${PROG2}.out ${PROG2}.c
${CC_MPI} $FLAGS -o ${PROG3}.out ${PROG3}.c

