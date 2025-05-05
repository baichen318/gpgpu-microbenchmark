NVCC := nvcc
PROJECT := sm2sm
FLAG := -arch=sm_90 -Xptxas -v

all:
	$(NVCC) $(FLAG) -o $(PROJECT) $(PROJECT).cu

clean:
	rm -rf *.o *.sw*
