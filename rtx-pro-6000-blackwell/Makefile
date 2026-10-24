CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_120a
NVCCFLAGS ?= -O3 -std=c++17 -gencode arch=compute_120a,code=$(ARCH) -lineinfo -Xcompiler -Wall
INCLUDES  := -Iinclude -I.
SRC       := $(wildcard src/bench_*.cu)
BINS      := $(patsubst src/%.cu,bin/%,$(SRC))

.PHONY: all clean

all: $(BINS)

bin/bench_power: src/bench_power.cu include/common.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ -l:libnvidia-ml.so.1

bin/%: src/%.cu include/common.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -rf bin
