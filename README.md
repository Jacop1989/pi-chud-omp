# pi-chud-omp

High-performance π (Pi) calculation using the **Chudnovsky series** with **binary splitting** and **OpenMP parallelism**.  
Implemented in C with **GMP** for high-precision arithmetic.

## Features


- Uses **Chudnovsky series** for fast convergence (~14.18 digits per term)
- **Binary splitting** for reduced computation complexity
- **OpenMP** multi-threading for parallel performance
- Supports outputting **exactly N decimal digits**
- Saves the result to `pi.txt`

## Requirements

- macOS with Homebrew  
- [GMP library](https://gmplib.org/) (GNU Multiple Precision Arithmetic Library)  
- [libomp](https://openmp.llvm.org/) for OpenMP support

## Install dependencies:


brew install gmp libomp

## Build


/usr/local/opt/llvm/bin/clang -std=c11 -O3 -march=native pi_chud_omp_plus.c -o pi_chud_omp_plus \
  -I"$(brew --prefix gmp)/include" -L"$(brew --prefix gmp)/lib" \
  -I"$(brew --prefix libomp)/include" -L"$(brew --prefix libomp)/lib" \
  -fopenmp -lomp -lgmp \
  -Wl,-rpath,"$(brew --prefix libomp)/lib" -Wl,-rpath,"$(brew --prefix gmp)/lib"

## RUN


Example: Calculate 1,000,000 digits of π using 4 threads

OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE ./pi_chud_omp_plus 1000000

## Output:


Prints the first 100 digits to the terminal
Shows performance statistics:





