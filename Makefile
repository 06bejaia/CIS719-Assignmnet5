CXX = g++  
NVCC = nvcc  
MPICXX = mpic++  
CUDAFLAGS = -O2  
CXXFLAGS = -O2 -fopenmp -pthread -fno-lto
ACCFLAGS = -fopenacc -fno-lto

# Executables  
all: floyd_native floyd_pthreads floyd_openmp floyd_mpi floyd_cuda floyd_thrust floyd_openacc  

# Native  
floyd_native: floyd_native.cpp  
	$(CXX) $(CXXFLAGS) floyd_native.cpp -o floyd_native  

# POSIX Threads  
floyd_pthreads: floyd_pthreads.cpp  
	$(CXX) $(CXXFLAGS) floyd_pthreads.cpp -o floyd_pthreads -lpthread  

# OpenMP (exclude MPI for OpenMP)  
floyd_openmp: floyd_openmp.cpp  
	$(CXX) $(CXXFLAGS) floyd_openmp.cpp -o floyd_openmp  

# MPI  
floyd_mpi: floyd_mpi.cpp  
	$(MPICXX) floyd_mpi.cpp -o floyd_mpi -lstdc++  

# CUDA  
floyd_cuda: floyd_cuda.cu  
	$(NVCC) $(CUDAFLAGS) floyd_cuda.cu -o floyd_cuda  

# Thrust (Thrust runs on CUDA)  
floyd_thrust: floyd_thrust.cu  
	$(NVCC) $(CUDAFLAGS) floyd_thrust.cu -o floyd_thrust  

# OpenACC  
floyd_openacc: floyd_openacc.cpp  
	$(CXX) $(ACCFLAGS) floyd_openacc.cpp -o floyd_openacc  

# Clean Rule  
clean:  
	rm -f floyd_native floyd_pthreads floyd_openmp floyd_mpi floyd_cuda floyd_thrust floyd_openacc  
