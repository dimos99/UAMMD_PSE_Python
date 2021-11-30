/*Raul P. Pelaez 2021. This code exposes in a class the UAMMD's PSE module.
  Allows to compute the hydrodynamic displacements of a group of particles due to thermal fluctuations and/or forces acting on them.
  See example.cpp for usage instructions.
  See example.py for usage from python.
 */
#include <uammd.cuh>
#include <Integrator/BDHI/BDHI_PSE.cuh>
#include <Integrator/BDHI/BDHI_FCM.cuh>
#include"uammd_interface.h"

namespace uammd_pse{
  using namespace uammd;
  using PSE = BDHI::PSE;
  using Kernel = BDHI::FCM_ns::Kernels::Gaussian;
  using KernelTorque = BDHI::FCM_ns::Kernels::GaussianTorque;
  using FCM = BDHI::FCM_impl<Kernel, KernelTorque>;      

  using Parameters = PSE::Parameters;

  struct Real3ToReal4{
    __host__ __device__ real4 operator()(real3 i){
      auto pr4 = make_real4(i);
      return pr4;
    }
  };


  Parameters toPSEParameters(PyParameters par){
    Parameters psepar;
    psepar.temperature = par.temperature;
    psepar.viscosity = par.viscosity;
    psepar.hydrodynamicRadius = par.hydrodynamicRadius;
    psepar.dt = 1;
    psepar.box = Box(make_real3(par.Lx, par.Ly, par.Lz));
    psepar.tolerance = par.tolerance;
    psepar.psi = par.psi;
    psepar.shearStrain = par.shearStrain;
    return psepar;  
  }

  FCM::Parameters toFCMParameters(PyParameters par){
    FCM::Parameters fcmpar;
    fcmpar.temperature = par.temperature;
    fcmpar.viscosity = par.viscosity;
    fcmpar.hydrodynamicRadius = par.hydrodynamicRadius;
    fcmpar.dt = 1;
    fcmpar.box = Box(make_real3(par.Lx, par.Ly, par.Lz));
    fcmpar.tolerance = par.tolerance;
    return fcmpar;  
  }

  struct UAMMD_PSE {
    using real = real;
    std::shared_ptr<System> sys;
    std::shared_ptr<ParticleData> pd;
    std::shared_ptr<PSE> pse;
    std::shared_ptr<FCM> fcm;
    thrust::device_vector<real> d_MF;
    thrust::device_vector<real3> tmp;
    int numberParticles;
    bool computeFluctuations = false;
    cudaStream_t st;
    UAMMD_PSE(PyParameters par, int numberParticles): numberParticles(numberParticles){   
      this->sys = std::make_shared<System>();
      this->pd = std::make_shared<ParticleData>(numberParticles, sys);
      auto pg = std::make_shared<ParticleGroup>(pd, sys, "All");
      if(par.psi <= 0){
	if(par.shearStrain != 0){
	  sys->log<System::EXCEPTION>("The non Ewald split FCM has no shear strain functionality");
	  throw std::runtime_error("Not implemented");
	}
	this->fcm = std::make_shared<FCM>(toFCMParameters(par));
      }
      else
	this->pse = std::make_shared<PSE>(pd, pg, sys, toPSEParameters(par));
      d_MF.resize(3*numberParticles);
      tmp.resize(numberParticles);
      CudaSafeCall(cudaStreamCreate(&st));
      if(par.temperature) this->computeFluctuations = true;
    }

    void Mdot(const real* h_pos,
	      const real* h_F,
	      real* h_MF){
      if(this->computeFluctuations){
	System::log<System::WARNING>("The Mdot function will not yield the correct fluctuations. Use computeHydrodynamicDisplacements instead");
      }
      uploadPosAndForceToUAMMD(h_pos, h_F);
      auto d_MF_ptr = (real3*)(thrust::raw_pointer_cast(d_MF.data()));
      this->computeMF(d_MF_ptr, st);
      thrust::copy(d_MF.begin(), d_MF.end(), h_MF);
    }

    void computeHydrodynamicDisplacements(const real* h_pos,
					  const real* h_F,
					  real* h_MF){
      uploadPosAndForceToUAMMD(h_pos, h_F);
      auto force = h_F?pd->getForce(access::gpu, access::read).begin():nullptr;
      if(pse){      
	auto d_MF_ptr = (real3*)(thrust::raw_pointer_cast(d_MF.data()));    
	pse->computeHydrodynamicDisplacements(force, d_MF_ptr, st);
	thrust::copy(d_MF.begin(), d_MF.end(), h_MF);
      }
      else if(fcm){
	auto pos = pd->getPos(access::gpu, access::read).begin();
	auto XT = fcm->computeHydrodynamicDisplacements(pos, force, nullptr, numberParticles, st);
	thrust::copy(XT.first.begin(), XT.first.end(), (real3*)h_MF);
      }
      else{
	sys->log<System::EXCEPTION>("PSE and FCM Modules are in an invalid state");
	throw std::runtime_error("[PSE]Invalid state");
      }
    }

    void MdotNearField(const real* h_pos,
		       const real* h_F,
		       real* h_MF){
      uploadPosAndForceToUAMMD(h_pos, h_F);
      auto d_MF_ptr = (real3*)(thrust::raw_pointer_cast(d_MF.data()));
      pse->computeMFNearField(d_MF_ptr, st);
      thrust::copy(d_MF.begin(), d_MF.end(), h_MF);
    }

    void MdotFarField(const real* h_pos,
		      const real* h_F,
		      real* h_MF){
      uploadPosAndForceToUAMMD(h_pos, h_F);
      auto d_MF_ptr = (real3*)(thrust::raw_pointer_cast(d_MF.data()));
      pse->computeMFFarField(d_MF_ptr, st);
      thrust::copy(d_MF.begin(), d_MF.end(), h_MF);
    }

    ~UAMMD_PSE(){
      cudaDeviceSynchronize();
      cudaStreamDestroy(st);
    }

    void clean(){
      cudaDeviceSynchronize();
      d_MF.clear();
      tmp.clear();
      fcm.reset();
      pse.reset();
      pd.reset();
      sys->finish();
      sys.reset();
    }

  private:
    void uploadPosAndForceToUAMMD(const real* h_pos, const real* h_F){
      auto pos = pd->getPos(access::location::gpu, access::mode::write);
      thrust::copy((real3*)h_pos, (real3*)h_pos + numberParticles, tmp.begin());
      thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(), pos.begin(), Real3ToReal4());
      if(h_F){
	auto forces = pd->getForce(access::location::gpu, access::mode::write);
	thrust::copy((real3*)h_F, (real3*)h_F + numberParticles, tmp.begin());
	thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(), forces.begin(), Real3ToReal4());
      }   
    }

    void computeMF(real3* d_MF_ptr, cudaStream_t st=0){
      if(pse)
	pse->computeMF(d_MF_ptr, st);
      else if(fcm){
	if(this->computeFluctuations){
	  System::log<System::WARNING>("The computeMF function will add fluctuations. Do not use this function in non-Ewald mode.");
	}
	auto pos = pd->getPos(access::gpu, access::read).begin();
	auto force = pd->getForce(access::gpu, access::read).begin();
	auto XT = fcm->computeHydrodynamicDisplacements(pos, force, nullptr, numberParticles, st);
	thrust::copy(XT.first.begin(), XT.first.end(), d_MF_ptr);
      }
      else{
	sys->log<System::EXCEPTION>("PSE and FCM Modules are in an invalid state");
	throw std::runtime_error("[PSE]Invalid state");
      }
    }

    void computeBdW(real3* d_MF_ptr, cudaStream_t st=0){
      if(pse)
	pse->computeBdW(d_MF_ptr, st);
      else if(fcm){
	auto pos = pd->getPos(access::gpu, access::read).begin();
	auto XT = fcm->computeHydrodynamicDisplacements(pos, nullptr, nullptr, numberParticles, st);
	thrust::copy(XT.first.begin(), XT.first.end(), d_MF_ptr);
      }
      else{
	sys->log<System::EXCEPTION>("PSE and FCM Modules are in an invalid state");
	throw std::runtime_error("[PSE]Invalid state");
      }
    }
  };

  UAMMD_PSE_Glue::UAMMD_PSE_Glue(PyParameters pypar, int numberParticles){
    pse = std::make_shared<UAMMD_PSE>(pypar, numberParticles);
  }

  void UAMMD_PSE_Glue::Mdot(const real* h_pos, const real* h_F, real* h_MF){
    pse->Mdot(h_pos, h_F, h_MF);
  }

  void UAMMD_PSE_Glue::MdotNearField(const real* h_pos, const real* h_F, real* h_MF){
    pse->MdotNearField(h_pos, h_F, h_MF);
  }

  void UAMMD_PSE_Glue::MdotFarField(const real* h_pos, const real* h_F, real* h_MF){
    pse->MdotFarField(h_pos, h_F, h_MF);
  }

  void UAMMD_PSE_Glue::computeHydrodynamicDisplacements(const real* h_pos,
							const real* h_F,
							real* h_MF){
    pse->computeHydrodynamicDisplacements(h_pos, h_F, h_MF);
  }

  void UAMMD_PSE_Glue::clean(){
    pse->clean();
  }
}
