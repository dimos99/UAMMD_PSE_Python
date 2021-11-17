/*Raul P. Pelaez 2021. This code exposes the UAMMD's PSE module to python.
  Allows to compute the hydrodynamic displacements of a group of particles due to thermal fluctuations and/or forces acting on them.  
  See example.py for usage from python.  
 */
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include "uammd_interface.h"

namespace py = pybind11;

struct UAMMD_PSE_Python {
  std::shared_ptr<UAMMD_PSE_Glue> pse;
  UAMMD_PSE_Python(PyParameters par, int numberParticles){
    pse = std::make_shared<UAMMD_PSE_Glue>(par, numberParticles);
  }

  void Mdot(py::array_t<real> h_pos,
	    py::array_t<real> h_F,
	    py::array_t<real> h_MF){
    pse->Mdot(h_pos.data(), h_F.data(), h_MF.mutable_data());
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
					py::array_t<real> h_MF){
    pse->computeHydrodynamicDisplacements(h_pos.data(), h_F.data(), h_MF.mutable_data());
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
    def("Mdot", &UAMMD_PSE_Python::Mdot, "Computes the hydrodynamic (deterministic and stochastic) displacements. If the forces are ommited only the stochastic part is computed. If the temperature is zero (default) the stochastic part is ommited.",
	"positions"_a,"forces"_a = py::array_t<real>(),"result"_a).
    def("MdotNearField", &UAMMD_PSE_Python::MdotNearField, "Computes only the far field contribution",
	"positions"_a,"forces"_a = py::array_t<real>(),"result"_a).
    def("MdotFarField", &UAMMD_PSE_Python::MdotFarField, "Computes only the deterministic part of the near field contribution",
	"positions"_a,"forces"_a= py::array_t<real>(),"result"_a).
    def("computeHydrodynamicDisplacements", &UAMMD_PSE_Python::computeHydrodynamicDisplacements,
	"Computes the hydrodynamic (deterministic and stochastic) displacements. If the forces are ommited only the stochastic part is computed. If the temperature is zero (default) the stochastic part is ommited.",
	"positions"_a,"forces"_a = py::array_t<real>(),"result"_a).
    def("clean", &UAMMD_PSE_Python::clean, "Frees any memory allocated by UAMMD");
  

    
  py::class_<PyParameters>(m, "PSEParameters").
    def(py::init([](real temperature,
		    real viscosity,
		    real hydrodynamicRadius,
		    real dt,
		    real Lx, real Ly, real Lz,
		    real tolerance,
		    real psi,
		    real shearStrain) {             
      auto tmp = std::unique_ptr<PyParameters>(new PyParameters);
      tmp->temperature = temperature;
      tmp->viscosity = viscosity;
      tmp->hydrodynamicRadius = hydrodynamicRadius;
      tmp->dt = dt;
      tmp->Lx = Lx;
      tmp->Ly = Ly;
      tmp->Lz= Lz;
      tmp->tolerance = tolerance;
      tmp->psi = psi;
      tmp->shearStrain = shearStrain;
      return tmp;	
    }),"temperature"_a = 0.0,"viscosity"_a  = 1.0,"hydrodynamicRadius"_a = 1.0,
	"dt"_a = 1.0,"Lx"_a = 0.0, "Ly"_a = 0.0, "Lz"_a = 0.0 ,"tolerance"_a = 1e-4,"psi"_a=1.0, "shearStrain"_a = 0.0).
    def_readwrite("temperature", &PyParameters::temperature).
    def_readwrite("viscosity", &PyParameters::viscosity).
    def_readwrite("hydrodynamicRadius", &PyParameters::hydrodynamicRadius).
    def_readwrite("dt", &PyParameters::dt).
    def_readwrite("tolerance", &PyParameters::tolerance).
    def_readwrite("psi", &PyParameters::psi).
    def_readwrite("Lx", &PyParameters::Lx).
    def_readwrite("Ly", &PyParameters::Ly).
    def_readwrite("Lz", &PyParameters::Lz).
    def_readwrite("shearStrain", &PyParameters::shearStrain).
    def("__str__", [](const PyParameters &p){
      return "temperature = "+std::to_string(p.temperature)+"\n"+
	"viscosity = " + std::to_string(p.viscosity) +"\n"+
	"hydrodynamicRadius = " + std::to_string(p.hydrodynamicRadius)+"\n"+
	"dt = " + std::to_string(p. dt)+ "\n" +
	"box (L = " + std::to_string(p.Lx) +
	"," + std::to_string(p.Ly) + "," + std::to_string(p.Lz) + ")\n"+
	"tolerance = " + std::to_string(p. tolerance)+ "\n" + 
	"psi = " + std::to_string(p. psi) + "\n" +	"shearStrain = " + std::to_string(p.shearStrain);
    });
    
    }
