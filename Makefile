CUDA_PATH    := /usr/local/cuda-12.9
CUDA_INC     := $(CUDA_PATH)/targets/x86_64-linux/include
CUDA_LIB     := $(CUDA_PATH)/targets/x86_64-linux/lib64

GPU_ARCH := -gencode arch=compute_75,code=sm_75
UAMMD_ROOT=uammd/
PYBIND_ROOT=pybind11/
PYTHON=python3
CXX = g++-12
NVCC = nvcc -ccbin=$(CXX) $(GPU_ARCH)

#In case you prefer to import with other name
MODULE_NAME=uammd
LIBRARY_NAME:=$(MODULE_NAME)$(shell $(PYTHON)-config --extension-suffix)

#UAMMD can be quite verbose, 5 shows only some messages at initialization/exit, 0 will only print critical errors, 15will print A LOT.
VERBOSITY=0
#Uncomment for double precision, UAMMD is compiled in single by default
#DOUBLEPRECISION=-DDOUBLE_PRECISION
#You can replace  intel's MKL by lapacke and cblas by removing -DUSE_MKL (in the include flags above) and linking with that instead
ifeq ($(MKLROOT),)
LAPACKINCLUDE=-I/usr/include/ -I/usr/include/x86_64-linux-gnu/
LAPACK_LIBS=-llapacke -lblas
else
LAPACKINCLUDE=-I$(MKLROOT)/include -L$(MKLROOT)/lib/intel64 -DUSE_MKL
LAPACK_LIBS=-lmkl_rt -lpthread -ldl
endif

INCLUDE_FLAGS_GPU= -I$(CUDA_INC) \
                   -I$(UAMMD_ROOT)/src \
                   -I$(UAMMD_ROOT)/src/third_party \
                   $(LAPACKINCLUDE)

INCLUDE_FLAGS= `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/

LDFLAGS_GPU= -L$(CUDA_LIB) -lcudart \
             -L/usr/lib/x86_64-linux-gnu -lcufft -lcublas \
             $(LAPACK_LIBS)


GPU_OPTIMIZATION= -O3
CPU_OPTIMIZATION= -O3

PYTHON_WRAPPER_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all: $(LIBRARY_NAME) example

uammd_wrapper.o: $(PYTHON_WRAPPER_DIR)/uammd_wrapper.cu $(PYTHON_WRAPPER_DIR)/uammd_interface.h
	$(NVCC) -w -std=c++14 -DMAXLOGLEVEL=$(VERBOSITY) $(GPU_OPTIMIZATION) $(DOUBLEPRECISION) $(INCLUDE_FLAGS_GPU) $(GPU_DEBUG) -Xcompiler "-fPIC -w" -c $< -o $@

uammd_python.o: $(PYTHON_WRAPPER_DIR)/uammd_python.cpp $(PYTHON_WRAPPER_DIR)/uammd_interface.h
	$(CXX) -std=c++14 -O3 $(DOUBLEPRECISION) -DMAXLOGLEVEL=$(VERBOSITY) $(CPU_DEBUG) $(CPU_OPTIMIZATION) -fPIC -w $(INCLUDE_FLAGS) -c $< -o $@

$(LIBRARY_NAME): uammd_wrapper.o uammd_python.o
	$(NVCC) $(DOUBLEPRECISION) $(GPU_OPTIMIZATION) -DMAXLOGLEVEL=$(VERBOSITY) $(GPU_DEBUG) -w -shared $^ -o $@ $(LDFLAGS_GPU)

example: example.cpp uammd_wrapper.o
	$(NVCC) $(DOUBLEPRECISION) $(GPU_OPTIMIZATION) $(GPU_DEBUG) $^ -o $@ $(LDFLAGS_GPU)
clean:
	rm -f $(LIBRARY_NAME) uammd_python.o uammd_wrapper.o example
