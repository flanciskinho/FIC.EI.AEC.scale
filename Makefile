#
# Makefile for MPI versions of Mandelbrot program
#
# "make" or "make all" to make all executables
# "make clean" to remove executables
#

OPT = -O3 -fopenmp -msse3
MPICC       = mpicc
CLINKER     = $(CC)
CFLAGS      = $(OPT) -Wall -pedantic -std=c99

MPICXX      = mpicxx
CCFLAGS     = $(OPT) -Wall -std=c++98 -Wno-long-long

LFLAGS      = -lm -lX11 -L/usr/X11R6/lib 

ALL = mpi_omp_sse
HFILES = 

.PHONY:  all
all:  $(ALL)

%: %.c $(HFILES)
	$(MPICC) -o $@ $(CFLAGS) $< $(LFLAGS)

.PHONY:  clean
clean:
	-rm $(ALL)

