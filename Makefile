
UAMMD_ROOT=uammd/
PYBIND_ROOT=pybind11/
PYTHON=python3
NVCC=nvcc

#UAMMD can be quite verbose, 5 shows only some messages at initialization/exit, 0 will only print critical errors, 15will print A LOT.
VERBOSITY=0
#Uncomment for double precision, UAMMD is compiled in single by default
#DOUBLEPRECISION=-DDOUBLE_PRECISION
#In caso you prefer to import with other name
MODULE_NAME=uammd
INCLUDE_FLAGS= -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/

#You can replace lapacke and cblas by intel's MKL using -DUSE_MKL and linking with that instead
LDFLAGS=-lcufft -lcublas -llapacke -lcblas

LIBRARY_NAME=$(MODULE_NAME)`$(PYTHON)-config --extension-suffix`
FILE=UAMMD_PSE_python_wrapper.cu
all:
	$(NVCC) -w -shared -std=c++11 -DMAXLOGLEVEL=$(VERBOSITY) $(DOUBLEPRECISION) $(INCLUDE_FLAGS) -Xcompiler "-fPIC -w"  $(FILE) -o $(LIBRARY_NAME) $(LDFLAGS)
clean:
	rm $(LIBRARY_NAME)
