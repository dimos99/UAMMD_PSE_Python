/*Raul P. Pelaez 2021-2022. This code exposes the UAMMD's PSE module to python.
  Allows to compute the hydrodynamic displacements of a group of particles due to thermal fluctuations and/or forces acting on them.  
  See example.py for usage from python.  
 */
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include "uammd_interface.h"
using namespace uammd_pse;
namespace py = pybind11;

struct UAMMD_PSE_Python {
  std::shared_ptr<UAMMD_PSE_Glue> pse;
  UAMMD_PSE_Python(PyParameters par, int numberParticles){
    pse = std::make_shared<UAMMD_PSE_Glue>(par, numberParticles);
  }

  void MdotNearField(py::array_t<real> h_pos,
		     py::array_t<real> h_F,
		     py::array_t<real> h_MF){
    pse->MdotNearField(h_pos.data(), h_F.data(), h_MF.mutable_data());
  }

  void MdotFarField(py::array_t<real> h_pos,
		    py::array_t<real> h_F,
		    py::array_t<real> h_MF){
    pse->MdotFarField(h_pos.data(), h_F.data(), h_MF.mutable_data());
  }

  void computeHydrodynamicDisplacements(py::array_t<real> h_pos,
					py::array_t<real> h_F,
					py::array_t<real> h_MF,
					real temperature, real prefactor){
    pse->computeHydrodynamicDisplacements(h_pos.data(), h_F.data(), h_MF.mutable_data(),
					  temperature, prefactor);
  }

  void setShearStrain(real newStrain){
    pse->setShearStrain(newStrain);
  }

  void clean(){
    pse->clean();
  }

};



using namespace pybind11::literals;

PYBIND11_MODULE(uammd, m) {
  m.doc() = "UAMMD PSE Python interface";
  py::class_<UAMMD_PSE_Python>(m, "UAMMD").
    def(py::init<PyParameters, int>(),"Parameters"_a, "numberParticles"_a).
    def("MdotNearField", &UAMMD_PSE_Python::MdotNearField, "Computes only the far field contribution",
	"positions"_a,"forces"_a = py::array_t<real>(),"result"_a).
    def("MdotFarField", &UAMMD_PSE_Python::MdotFarField, "Computes only the deterministic part of the near field contribution",
	"positions"_a,"forces"_a= py::array_t<real>(),"result"_a).
    def("computeHydrodynamicDisplacements", &UAMMD_PSE_Python::computeHydrodynamicDisplacements,
	"Computes the hydrodynamic (deterministic and/or stochastic) displacements. If the forces are ommited only the stochastic part is computed. If the temperature is zero (default) the stochastic part is ommited.",
	"positions"_a,"forces"_a = py::array_t<real>(),"result"_a, "temperature"_a = 0, "prefactor"_a = 0).
    def("setShearStrain", &UAMMD_PSE_Python::setShearStrain, "Sets a new value for the shear strain (only in PSE mode).",
	"newShearStrain"_a).
    def("clean", &UAMMD_PSE_Python::clean, "Frees any memory allocated by UAMMD");
    
  py::class_<PyParameters>(m, "PSEParameters").
    def(py::init([](real viscosity,
		    real hydrodynamicRadius,
		    real Lx, real Ly, real Lz,
		    real tolerance,
		    real psi,
		    real shearStrain) {             
      auto tmp = std::unique_ptr<PyParameters>(new PyParameters);
      tmp->viscosity = viscosity;
      tmp->hydrodynamicRadius = hydrodynamicRadius;
      tmp->Lx = Lx;
      tmp->Ly = Ly;
      tmp->Lz= Lz;
      tmp->tolerance = tolerance;
      tmp->psi = psi;
      tmp->shearStrain = shearStrain;
      return tmp;	
    }),"viscosity"_a  = 1.0,"hydrodynamicRadius"_a = 1.0,
	"Lx"_a = 0.0, "Ly"_a = 0.0, "Lz"_a = 0.0 ,"tolerance"_a = 1e-4,"psi"_a=1.0, "shearStrain"_a = 0.0).
    def_readwrite("viscosity", &PyParameters::viscosity).
    def_readwrite("hydrodynamicRadius", &PyParameters::hydrodynamicRadius).
    def_readwrite("tolerance", &PyParameters::tolerance).
    def_readwrite("psi", &PyParameters::psi).
    def_readwrite("Lx", &PyParameters::Lx).
    def_readwrite("Ly", &PyParameters::Ly).
    def_readwrite("Lz", &PyParameters::Lz).
    def_readwrite("shearStrain", &PyParameters::shearStrain).
    def("__str__", [](const PyParameters &p){
      return "viscosity = " + std::to_string(p.viscosity) +"\n"+
	"hydrodynamicRadius = " + std::to_string(p.hydrodynamicRadius)+"\n"+
	"box (L = " + std::to_string(p.Lx) +
	"," + std::to_string(p.Ly) + "," + std::to_string(p.Lz) + ")\n"+
	"tolerance = " + std::to_string(p. tolerance)+ "\n" + 
	"psi = " + std::to_string(p. psi) + "\n" +	"shearStrain = " + std::to_string(p.shearStrain);
    });
    
    }
