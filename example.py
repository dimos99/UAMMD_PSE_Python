#Raul P. Pelaez 2021-2022. Example usage of the UAMMD's PSE module (see uammd_python.cpp for more info on the wrapper)
#Computes the hydrodynamic displacements of a group of particles due to thermal fluctuations and/or forces acting on them.
import uammd
import numpy as np

numberParticles = 20000;
L=120.0;
par = uammd.PSEParameters(psi=0.3, viscosity=1.0, hydrodynamicRadius=1.0, tolerance=1e-4,
                          Lx=L, Ly=L, Lz=L, shearStrain=0.0);
pse = uammd.UAMMD(par, numberParticles);

#pse.Mdot assumes interleaved positions and forces, that is x1,y1,z1,x2,y2,z2,...
np.random.seed(1234)
positions = np.array((np.random.rand(3*numberParticles)-0.5)*L, np.float32);
print(np.reshape(positions, (numberParticles, 3))[0:3]);
forces = np.array((np.random.rand(3*numberParticles)-0.5), np.float32);
#It is really important that the result array has the same floating precision as the compiled uammd, otherwise
# python will just silently pass by copy and the results will be lost
MF=np.zeros(3*numberParticles, np.float32);
pse.computeHydrodynamicDisplacements(positions, forces, temperature = 0, prefactor = 0, MF)
# We can also change the shear strain at any point
pse.setShearStrain(1.0)
print(MF)

