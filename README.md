
### UAMMDs Brownian Hydrodynamics Positively Split Ewald module wrapper for python.

UAMMD: [https://github.com/RaulPPelaez/uammd]()  

This code allows to call UAMMD's PSE module from a python script to compute the product of the mobility with a certain vector (usually forces acting on particle positions).  
You can read about PSE here: [https://github.com/RaulPPelaez/UAMMD/wiki/BDHI-PSE]()   

### Requirements

You need [UAMMD](https://github.com/RaulPPelaez/uammd) and [pybind11](https://github.com/pybind/pybind11), both of which are included in this repository as git submodules. Remember to clone recursively with:  
```shell
git clone --recursive https://github.com/RaulPPelaez/UAMMD_PSE_Python
```

You need the CUDA toolkit installed, I tested up to the latest version at the moment (CUDA 11).  

The PSE module uses cufft, cublas, lapacke and cblas which need to be available during compilation. CuFFT and CuBLAS should be available as part of the CUDA installation.  

Lapacke and cblas are not actually needed, they are only used for to compute fluctuations, a capability which this code does not expose. Still the code is included so it must be compiled with them. Both can be replaced by intels MKL libraries via Makefile, you could even replace them with a stub given that no function in these libraries will be actually called.


### Compilation

Just run `make` and thats it. Notice that you may need to modify it with the appropiate paths of your systems (mainly the location of nvcc and python).  
The Makefile uses the `python-config` utility and assumes that is called `$(PYTHON)-config`, change it otherwise.  
You may enable debug mode by adding `-DUAMMD_DEBUG -g -G` to the compilation line, also increasing the verbosity in the makefile will help track down any runtime issues.  
The Makefile also assumes `nvcc` is in the PATH but you can easily change it to point to wherever it is.   


### Example
Do not try to modify the parameters after you have created the UAMMD object as it will have no effect whatsoever in it.  
Mdot expects positions and forces to be in a one dimensional array with interleaved pattern such that positions = [x1, y1, z1, x2, y2, z2...]. Same goes for the output, which must have the correct size (3*N) when passed to Mdot.  
```python
import uammd
numberParticles = 20000;
L=120.0;
par = uammd.PSEParameters(psi=0.3, viscosity=1.0, hydrodynamicRadius=1.0, tolerance=1e-4, box=uammd.Box(L,L,L));
pse = uammd.UAMMD(par, numberParticles);
...
pse.Mdot(positions, forces, MF)

```
