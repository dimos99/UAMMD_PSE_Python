#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <uammd.cuh>
#include <Integrator/BDHI/BDHI_PSE.cuh>
namespace py = pybind11;
using PSE = uammd::BDHI::PSE;
using Parameters = PSE::Parameters;

struct Real3ToReal4{
  __host__ __device__ uammd::real4 operator()(uammd::real3 i){
    auto pr4 = uammd::make_real4(i);
    return pr4;
  }
};

struct UAMMD {
  using real = uammd::real;
  std::shared_ptr<uammd::System> sys;
  std::shared_ptr<uammd::ParticleData> pd;
  std::shared_ptr<uammd::BDHI::PSE> pse;
  thrust::device_vector<real> d_MF;
  thrust::device_vector<uammd::real3> tmp;
  int numberParticles;
  cudaStream_t st;
  UAMMD(Parameters par, int numberParticles): numberParticles(numberParticles){
    this->sys = std::make_shared<uammd::System>();
    this->pd = std::make_shared<uammd::ParticleData>(numberParticles, sys);
    auto pg = std::make_shared<uammd::ParticleGroup>(pd, sys, "All");
    this->pse = std::make_shared<PSE>(pd, pg, sys, par);
    d_MF.resize(3*numberParticles);
    tmp.resize(numberParticles);
    CudaSafeCall(cudaStreamCreate(&st));
  }

  void Mdot(py::array_t<real> h_pos,
	    py::array_t<real> h_F,
	    py::array_t<real> h_MF){
    {
      auto pos = pd->getPos(uammd::access::location::gpu, uammd::access::mode::write);
      thrust::copy((uammd::real3*)h_pos.data(), (uammd::real3*)h_pos.data() + numberParticles, tmp.begin());
      thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(), pos.begin(), Real3ToReal4());
      auto forces = pd->getForce(uammd::access::location::gpu, uammd::access::mode::write);
      thrust::copy((uammd::real3*)h_F.data(), (uammd::real3*)h_F.data() + numberParticles, tmp.begin());
      thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(), forces.begin(), Real3ToReal4());
    }
    auto d_MF_ptr = (uammd::real3*)(thrust::raw_pointer_cast(d_MF.data()));
    pse->computeMF(d_MF_ptr, st);
    thrust::copy(d_MF.begin(), d_MF.end(), h_MF.mutable_data());
  }
  
  ~UAMMD(){
    cudaDeviceSynchronize();
    cudaStreamDestroy(st);
  }
};



using namespace pybind11::literals;

PYBIND11_MODULE(uammd, m) {
  m.doc() = "UAMMD PSE Python interface";
  py::class_<UAMMD>(m, "UAMMD").
    def(py::init<Parameters, int>(),"Parameters"_a, "numberParticles"_a).
    def("Mdot", &UAMMD::Mdot, "Computes the product of the Mobility tensor with a provided array",
	"positions"_a,"forces"_a,"result"_a);
  
  py::class_<uammd::Box>(m, "Box").
    def(py::init<uammd::real>()).
    def(py::init([](uammd::real x, uammd::real y, uammd::real z) {
      return std::unique_ptr<uammd::Box>(new uammd::Box(uammd::make_real3(x,y,z)));
    }));

  py::class_<Parameters>(m, "PSEParameters").
    def(py::init([](uammd::real temperature,
		    uammd::real viscosity,
		    uammd::real hydrodynamicRadius,
		    uammd::real dt,
		    uammd::Box  box,
		    uammd::real tolerance,
		    uammd::real psi,
		    uammd::real shearStrain) {             
      auto tmp = std::unique_ptr<Parameters>(new Parameters);
      tmp->temperature = temperature;
      tmp->viscosity = viscosity;
      tmp->hydrodynamicRadius = hydrodynamicRadius;
      tmp->dt = dt;
      tmp->box = box;
      tmp->tolerance = tolerance;
      tmp->psi = psi;
      tmp->shearStrain = shearStrain;
      return tmp;	
    }),"temperature"_a = 0.0,"viscosity"_a  = 1.0,"hydrodynamicRadius"_a = 1.0,
	"dt"_a = 0.0,"box"_a = uammd::Box(),"tolerance"_a = 1e-4,"psi"_a=1.0, "shearStrain"_a = 0.0).
    def_readwrite("temperature", &Parameters::temperature).
    def_readwrite("viscosity", &Parameters::viscosity).
    def_readwrite("hydrodynamicRadius", &Parameters::hydrodynamicRadius).
    def_readwrite("dt", &Parameters::dt).
    def_readwrite("tolerance", &Parameters::tolerance).
    def_readwrite("psi", &Parameters::psi).
    def_readwrite("box", &Parameters::box).
    def_readwrite("shearStrain", &Parameters::shearStrain).
    def("__str__", [](const Parameters &p){
      return "temperature = "+std::to_string(p.temperature)+"\n"+
	"viscosity = " + std::to_string(p.viscosity) +"\n"+
	"hydrodynamicRadius = " + std::to_string(p.hydrodynamicRadius)+"\n"+
	"dt = " + std::to_string(p. dt)+ "\n" +
	"box (L = " + std::to_string(p.box.boxSize.x) +
	"," + std::to_string(p.box.boxSize.y) + "," + std::to_string(p.box.boxSize.z) + ")\n"+
	"tolerance = " + std::to_string(p. tolerance)+ "\n" + 
	"psi = " + std::to_string(p. psi) + "\n" +
	"shearStrain = " + std::to_string(p.shearStrain);
    });
    
    }
