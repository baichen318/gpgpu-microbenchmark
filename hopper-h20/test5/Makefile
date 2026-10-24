NVCC := nvcc
PROJECT := load
FLAG := -O3
LIB := -lcudart -lnvidia-ml

all:
	$(NVCC) $(FLAG) -o $(PROJECT) $(PROJECT).cu $(LIB)

clean:
	rm -rf *.o *.sw*
