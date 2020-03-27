#Makefile
compile: 
	mpicc src/dijkstra.c -o  parallel
	@echo "Executable file is ready"

run:
	@echo "Running with 100 thread"
	mpirun -np 100 --oversubscribe parallel ${node}

all: compile run