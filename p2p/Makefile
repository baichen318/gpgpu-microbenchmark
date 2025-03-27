NVCC := nvcc
PROJECT := p2p
FLAG := -O3
LIB := -lcudart

all:
	$(NVCC) $(FLAG) -o $(PROJECT) $(PROJECT).cu $(LIB)

clean:
	rm -rf *.o *.sw*
