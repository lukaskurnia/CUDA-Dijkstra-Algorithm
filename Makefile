# Makefile
# Running with 100 node
default: compile run 

compile: 
	nvcc src/dijkstra.cu -o cuda.out
	@echo "Executable file is ready"

run:
	@echo "Running..."
	./cuda.out ${node}
