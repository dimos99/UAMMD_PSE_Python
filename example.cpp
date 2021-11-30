/* Raul P. Pelaez 2021. Example usage of the c++ PSE interface. See example.py for the python interface
   Computes the hydrodynamic displacements for a group of randomly placed particles with forces acting on them.
   Also includes fluctuations if temperature !=0
 */
#include"uammd_interface.h"
#include<vector>
#include<iostream>
#include<random>
#include<algorithm>
using namespace uammd_pse;
using PSE = UAMMD_PSE_Glue;
using PSEParameters = PyParameters;
using namespace std;

//Some arbitrary parameters
auto createParameters(){
  PSEParameters par;
  par.temperature = 0.1;
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.Lx = 120.0;
  par.Ly = 120.0;
  par.Lz = 120.0;
  par.tolerance = 1e-4;
  par.psi = 0.3;
  par.shearStrain = 0.0;
  return par;
}

//Fills a vector of "size" elements with uniform random numbers between -var and +var 
auto createRandomVector(int size, real var){
  static std::mt19937 gen(0xBADA55D00D);
  std::vector<real> vec(size);
  std::uniform_real_distribution<real> dis(-0.5, 0.5);
  std::generate(vec.begin(), vec.end(), [&](){return dis(gen)*var;});
  return vec;
}

int main(){
  auto par = createParameters();
  int numberParticles = 20000;
  PSE pse(par, numberParticles);
  auto pos = createRandomVector(3*numberParticles, par.Lx);
  auto forces = createRandomVector(3*numberParticles, par.Lx);
  std::vector<real> MF(3*numberParticles, 0);
  pse.Mdot(pos.data(), forces.data(), MF.data());
  //Each part can be requested independently
  // pse.MdotNearField(pos.data(), forces.data(), MF.data());
  // pse.MdotFarField(pos.data(), forces.data(), MF.data());
  for(int i = 0; i< 3*10; i++){
    std::cout<<MF[i]<<" ";
    if(i%3==2) std::cout<<std::endl;
  }
  return 0;
}
