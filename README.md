
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

### Compilation

#### Using Makefile (legacy)
Just run `make` and that's it. You may need to modify the Makefile for your system (e.g., paths to nvcc, python, etc.).

#### Using CMake (recommended)
1. Create a build directory and configure the project:
   ```sh
   mkdir build && cd build
   cmake ..
   make -j4
   ```
2. Install the Python module and example executable:
   ```sh
   make install
   ```
   This will copy the Python module (`uammd`) directly into your current Python environment's site-packages directory, so you can import it from anywhere without modifying `sys.path` or using `PYTHONPATH`. You might need to run the install command with `sudo` (i.e., `sudo make install`).

#### Notes
- The Makefile builds the Python module in the project root, while CMake builds it in `build/` and installs it to your Python environment.
- You can still use the Makefile if you prefer, but CMake is more portable and robust.

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
pse.computeHydrodynamicDisplacements(positions, forces, MF, temperature = 1.0, prefactor = 1.0)

```
The function ```computeHydrodynamicDisplacements``` computes the full hydrodynamic displacements (including the deterministic and stochastic ones). If the "forces" argument is ommited only the stochastic contribution is computed. On the other hand, setting temperature=0 will result in only the deterministic part being computed. The prefactor affects the stochastic part, the result MF will be ```MF = Mobility*forces + prefactor*sqrt(2*temperature*Mobility)*dW)```.  

PSE uses Ewald splitting, with a Near and Far field contribution. Each part of the deterministic computation can be computed separatedly with  

```python
pse.MdotNearField(positions, forces, MF)
pse.MdotFarField(positions, forces, MF)

```

Finally, the shearStrain parameter allows to use sheared boxes.  

### C++ interface  

The module can also be called from C++ by including uammd_interface.h. See example.cpp.  
