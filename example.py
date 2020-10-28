import uammd
import numpy as np

numberParticles = 20000;
L=120.0;

par = uammd.PSEParameters(psi=0.3, viscosity=1.0, hydrodynamicRadius=1.0, tolerance=1e-4, box=uammd.Box(L,L,L));
pse = uammd.UAMMD(par, numberParticles);

#pse.Mdot assumes interleaved positions and forces, that is x1,y1,z1,x2,y2,z2,...
positions = np.array((np.random.rand(3*numberParticles)-0.5)*L, np.float32);
forces = np.array((np.random.rand(3*numberParticles)-0.5), np.float32);
#It is really important that the result array has the same floating precision as the compiled uammd, otherwise
# python will just silently pass by copy and the results will be lost
MF=np.zeros(3*numberParticles, np.float32);
pse.Mdot(positions, forces, MF)
print(MF)

