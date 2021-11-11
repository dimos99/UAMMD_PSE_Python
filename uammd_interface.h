/*Raul P. Pelaez 2021.
  An interface code between uammd_wrapper.cu and uammd_python.cpp.
 */
#include<string>
#include<memory>

//This is in order not to use any UAMMD related includes here.
//Instead of using uammd::real I have to re define real here.
#ifndef DOUBLE_PRECISION
using real = float;
#else
using real = double;
#endif

//This function returns either 'single' or 'double' according to the UAMMD's compiled precision.
namespace uammd_wrapper{
  std::string getPrecision();
}

struct PyParameters{
  real temperature;
  real viscosity;
  real hydrodynamicRadius;
  real dt = 1;
  real Lx, Ly, Lz;
  real tolerance;
  real psi;
  real shearStrain;
};

class UAMMD_PSE;
class UAMMD_PSE_Glue{
  std::shared_ptr<UAMMD_PSE> pse;
public:

  UAMMD_PSE_Glue(PyParameters pypar, int numberParticles);

  void Mdot(const real* h_pos, const real* h_F, real* h_MF);

  void MdotNearField(const real* h_pos, const real* h_F, real* h_MF);

  void MdotFarField(const real* h_pos, const real* h_F, real* h_MF);

  void computeHydrodynamicDisplacements(const real* h_pos, const real* h_F, real* h_MF);

  void clean();
};
