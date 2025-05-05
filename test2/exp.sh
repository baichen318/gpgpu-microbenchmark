#!/bin/bash
# SM Connectivity Profiler

# Configurations
PROJECT="sm2sm"
BIN_DIR="./bin"
LOG_DIR="./logs"

# Create directories.
mkdir -p ${BIN_DIR} ${LOG_DIR}

compile_project() {
    echo "Compiling with stride=$1 ..."
    nvcc -arch=sm_90 \
         -Xptxas="-v" \
         -o ${BIN_DIR}/${PROJECT}_$1 \
         ${PROJECT}.cu 2>&1 | tee ${LOG_DIR}/build_$1.log
}

run_profile() {
    local stride=$1
    echo "[$(date +%T)] Profiling stride=$stride"
    
    nsys profile \
        --stats=true \
        --trace cuda,nvtx \
        ${BIN_DIR}/${PROJECT}_$stride $stride \
        > ${LOG_DIR}/stride_${stride}.log 2>&1
}

main() {
    # We test from stride 1 to stride 18.
    for stride in {1..18}; do
        compile_project $stride
        run_profile $stride
        sleep 1
    done
    
    echo "All stride profiles saved to ${LOG_DIR}"
    nvidia-smi --query | grep "GPU Util"
}

main "$@"