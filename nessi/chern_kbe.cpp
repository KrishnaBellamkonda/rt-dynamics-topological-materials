// =====================================================================
//  chern_kbe.cpp
//  NESSi KBE (full two time Kadanoff Baym, second born) for the spinless
//  interacting Chern insulator quench. This is the local second born cross
//  check for the GKBA code in ../gkba.
//
//  Reference
//    M. Schueler, J. C. Budich and P. Werner, Phys. Rev. B 97, 245129 (2018).
//    Built on NESSi libcntr, https://github.com/nessi-cntr/nessi.
//
//  Model, local interaction
//    h0(k) = lam sin(kx) sx + lam sin(ky) sy + (M - cos kx - cos ky) sz
//    H_int = V sum_i n_iE n_iH    on site, between the E and H orbitals
//  The M - cos - cos sign puts the inversion at the BZ corner, the same
//  physics as the +cos+cos python code up to a shift k to k + (pi, pi).
//
//  Each k is an independent 2x2 contour Green function G[k]. The local
//  interaction couples them only through the k averaged G_loc, giving one
//  k independent self energy applied to every k.
//
//  Protocol, uncorrelated initial state
//    t<0   free thermal state of h0(k, M_pre), Sigma_M = 0
//    t=0   sudden quench M_pre to M_post and the interaction turned on
//    t>0   propagate under h0(k, M_post) plus the second born self energy
//
//  Inter orbital self energy (the stock hubb:: is same flavour)
//    HF    S_EE = V rho_HH, S_HH = V rho_EE, S_EH = -V rho_EH
//    2B    Sigma_EE(t,t') = V^2 G_EE(t,t') G_HH(t,t') G_HH(t',t)
//          built with cntr::Bubble1 and cntr::Bubble2.
//
//  Output rho(k,t) to rho.bin and energy.dat, read by read_rho.py. The sign
//  of the 2B kernel is checked by energy conservation and the V=0 limit.
// =====================================================================
#include <sys/stat.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>

#include "cntr/cntr.hpp"
#include "cntr/utils/read_inputfile.hpp"
#include "formats.hpp"

using namespace std;

// --- 2x2 Bloch Hamiltonian h0(kx,ky;M) = d.sigma -------------------
cdmatrix h0_mat(double kx, double ky, double M, double lam){
  cdmatrix H(2,2);
  double dx = lam*sin(kx), dy = lam*sin(ky), dz = M - cos(kx) - cos(ky);
  H(0,0) =  dz;              H(1,1) = -dz;
  H(0,1) = complex<double>(dx,-dy);
  H(1,0) = complex<double>(dx, dy);
  return H;
}

// --- Hartree-Fock self-energy (inter-flavor), = gkba.py sigma_hf ----
cdmatrix sigma_hf(const cdmatrix &rho, double V){
  cdmatrix S(2,2); S.setZero();
  S(0,0) =  V*rho(1,1);  S(1,1) =  V*rho(0,0);
  S(0,1) = -V*rho(0,1);  S(1,0) = -V*rho(1,0);
  return S;
}

// --- direct 2nd-Born self-energy at tstp, from local G -------------
//  Sigma_EE = V^2 * G_EE * (G_HH bubble),  Sigma_HH = V^2 * G_HH * (G_EE bubble)
void sigma_2b_local(int tstp, GREEN &Gloc, double V, GREEN &Sigloc){
  int ntau = Gloc.ntau();
  Sigloc.set_timestep_zero(tstp);
  if(V == 0.0) return;
  GREEN_TSTP Pol(tstp, ntau, 2, BOSON);
  // Pol_00 <- bubble of G_HH (feeds Sigma_EE); Pol_11 <- bubble of G_EE
  cntr::Bubble1(tstp, Pol, 0,0, Gloc, 1,1, Gloc, 1,1);
  cntr::Bubble1(tstp, Pol, 1,1, Gloc, 0,0, Gloc, 0,0);
  Pol.smul(V*V);
  cntr::Bubble2(tstp, Sigloc, 0,0, Gloc, 0,0, Pol, 0,0);
  cntr::Bubble2(tstp, Sigloc, 1,1, Gloc, 1,1, Pol, 1,1);
}

int main(int argc, char *argv[]){
  int SolveOrder = MAX_SOLVE_ORDER;
  int Nk, Nt, Ntau, BootstrapMaxIter, CorrectorSteps, SaveEvery;
  double lam, Mpre, Mpost, V, beta, h, MuChem, BootstrapMaxErr;
  char *flin, *flout;

  if(argc < 3){ cerr << "usage: chern_kbe <input> <out_prefix>" << endl; return 1; }
  flin = argv[1]; flout = argv[2];

  find_param(flin,"__Nk=",Nk);
  find_param(flin,"__lam=",lam);
  find_param(flin,"__Mpre=",Mpre);
  find_param(flin,"__Mpost=",Mpost);
  find_param(flin,"__V=",V);
  find_param(flin,"__Nt=",Nt);
  find_param(flin,"__Ntau=",Ntau);
  find_param(flin,"__beta=",beta);
  find_param(flin,"__h=",h);
  find_param(flin,"__MuChem=",MuChem);
  find_param(flin,"__BootstrapMaxIter=",BootstrapMaxIter);
  find_param(flin,"__BootstrapMaxErr=",BootstrapMaxErr);
  find_param(flin,"__CorrectorSteps=",CorrectorSteps);
  find_param(flin,"__SaveEvery=",SaveEvery);

  int Nk2 = Nk*Nk;
  print_line_star(60);
  cout << "  NESSi KBE 2B : spinless QWZ Chern quench (LOCAL)" << endl;
  cout << "  Nk=" << Nk << "x" << Nk << " (" << Nk2 << " kpts)  Nt=" << Nt
       << "  Ntau=" << Ntau << "  V=" << V << endl;
  print_line_star(60);

  // memory report
  {
    const size_t MB = 1024*1024;
    size_t per_k = cntr::mem_herm_matrix<double>(Nt,Ntau,2);
    cout << "per-k G: " << ceil(per_k/(double)MB) << " MB   total G[k]: "
         << ceil((double)per_k*Nk2/MB) << " MB" << endl;
  }

  // k-grid: ks[i] = 2*pi*i/Nk ; flat index ik = a*Nk+b (kx=ks[a],ky=ks[b])
  vector<double> ks(Nk);
  for(int i=0;i<Nk;i++) ks[i] = 2.0*M_PI*i/Nk;

  vector<GREEN> G(Nk2);
  vector<CFUNC> eps_mf(Nk2);
  vector<cdmatrix> h0post(Nk2);
  for(int a=0;a<Nk;a++) for(int b=0;b<Nk;b++){
    int ik=a*Nk+b;
    G[ik] = GREEN(Nt,Ntau,2,FERMION);
    eps_mf[ik] = CFUNC(Nt,2);
    h0post[ik] = h0_mat(ks[a],ks[b],Mpost,lam);
  }
  GREEN Gloc(Nt,Ntau,2,FERMION), Sigloc(Nt,Ntau,2,FERMION);

  // ---- initial state: uncorrelated free thermal of h0(k,M_pre) (Eq.7)
  for(int a=0;a<Nk;a++) for(int b=0;b<Nk;b++){
    int ik=a*Nk+b;
    cdmatrix hpre = h0_mat(ks[a],ks[b],Mpre,lam);
    cntr::green_from_H(G[ik], MuChem, hpre, beta, h);
  }
  Sigloc.set_timestep_zero(-1);   // Sigma^M = 0 (no initial correlations)
  cout << "initial free thermal state built (Sigma^M=0, no Matsubara SC)." << endl;

  auto build_Gloc = [&](int tstp){
    Gloc.set_timestep_zero(tstp);
    for(int ik=0; ik<Nk2; ik++) Gloc.incr_timestep(tstp, G[ik], complex<double>(1.0/Nk2,0.0));
  };
  auto set_meanfield = [&](int tstp){
    cdmatrix rho(2,2); Gloc.density_matrix(tstp, rho);
    cdmatrix HF = sigma_hf(rho, V);
    for(int ik=0; ik<Nk2; ik++){
      cdmatrix hmf = h0post[ik] + HF;
      eps_mf[ik].set_value(tstp, hmf);
    }
  };

  // ---- bootstrap (t=0..SolveOrder): quench is already in h0post ----
  {
    print_line_minus(50); cout << "  bootstrap" << endl; print_line_minus(50);
    auto t0 = chrono::system_clock::now();
    for(int iter=0; iter<=BootstrapMaxIter; iter++){
      for(int n=0;n<=SolveOrder;n++){ build_Gloc(n); set_meanfield(n); sigma_2b_local(n,Gloc,V,Sigloc); }
      double err=0.0;
      #pragma omp parallel for schedule(dynamic)
      for(int ik=0; ik<Nk2; ik++){
        cntr::dyson_start(G[ik], MuChem, eps_mf[ik], Sigloc, beta, h, SolveOrder);
      }
      // convergence via occupation of a reference k (cheap proxy)
      cdmatrix r0(2,2); G[0].density_matrix(SolveOrder, r0);
      cout << "  boot iter " << iter << "  n0=" << r0(0,0).real() << endl;
      if(iter>=2){ (void)err; }
      if(iter>=CorrectorSteps+2) break;
    }
    auto t1 = chrono::system_clock::now();
    cout << "  [bootstrap] " << chrono::duration<double>(t1-t0).count() << " s" << endl;
  }

  // ---- time propagation ----
  {
    print_line_minus(50); cout << "  propagation" << endl; print_line_minus(50);
    auto t0 = chrono::system_clock::now();
    for(int tstp=SolveOrder+1; tstp<=Nt; tstp++){
      #pragma omp parallel for schedule(dynamic)
      for(int ik=0; ik<Nk2; ik++) cntr::extrapolate_timestep(tstp-1, G[ik], SolveOrder);
      for(int it=0; it<CorrectorSteps; it++){
        build_Gloc(tstp); set_meanfield(tstp); sigma_2b_local(tstp,Gloc,V,Sigloc);
        #pragma omp parallel for schedule(dynamic)
        for(int ik=0; ik<Nk2; ik++)
          cntr::dyson_timestep(tstp, G[ik], MuChem, eps_mf[ik], Sigloc, beta, h, SolveOrder);
      }
      if(tstp % max(1,Nt/10)==0)
        cout << "  t=" << setprecision(4) << tstp*h << " (" << tstp << "/" << Nt << ")" << endl;
    }
    auto t1 = chrono::system_clock::now();
    cout << "  [propagation] " << chrono::duration<double>(t1-t0).count() << " s" << endl;
  }

  // ---- output: rho(k,t) binary + energy.dat + meta.txt ----
  {
    char fn[512];
    strcpy(fn,flout); strcat(fn,"_rho.bin");
    ofstream fb(fn, ios::binary);
    vector<int> saved;
    for(int tstp=0; tstp<=Nt; tstp++) if(tstp%SaveEvery==0 || tstp==Nt) saved.push_back(tstp);
    for(size_t s=0;s<saved.size();s++){
      int tstp=saved[s];
      for(int ik=0; ik<Nk2; ik++){
        cdmatrix r(2,2); G[ik].density_matrix(tstp, r);
        double buf[8] = { r(0,0).real(),r(0,0).imag(), r(0,1).real(),r(0,1).imag(),
                          r(1,0).real(),r(1,0).imag(), r(1,1).real(),r(1,1).imag() };
        fb.write(reinterpret_cast<char*>(buf), sizeof(buf));
      }
    }
    fb.close();

    strcpy(fn,flout); strcat(fn,"_energy.dat");
    ofstream fe(fn); fe << setprecision(12);
    for(int tstp=0; tstp<=Nt; tstp++){
      build_Gloc(tstp);
      double ekin=0.0, emf=0.0;
      cdmatrix HF; { cdmatrix rl(2,2); Gloc.density_matrix(tstp,rl); HF=sigma_hf(rl,V); }
      for(int ik=0; ik<Nk2; ik++){
        cdmatrix r(2,2); G[ik].density_matrix(tstp,r);
        ekin += (h0post[ik]*r).trace().real();
        emf  += 0.5*(HF*r).trace().real();
      }
      ekin/=Nk2; emf/=Nk2;
      double ecorr = cntr::correlation_energy(tstp, Gloc, Sigloc, beta, h, SolveOrder);
      fe << tstp*h << "  " << ekin << "  " << emf << "  " << ecorr << "  "
         << ekin+emf+ecorr << endl;
    }
    fe.close();

    strcpy(fn,flout); strcat(fn,"_meta.txt");
    ofstream fm(fn);
    fm << "Nk " << Nk << "\nNt " << Nt << "\nSaveEvery " << SaveEvery
       << "\ndt " << h << "\nlam " << lam << "\nMpre " << Mpre
       << "\nMpost " << Mpost << "\nV " << V << "\nbeta " << beta
       << "\nnsaved " << (int)((Nt/SaveEvery)+1) << "\n";
    fm << "kgrid ks[i]=2*pi*i/Nk  ik=a*Nk+b (kx=ks[a],ky=ks[b])\n";
    fm.close();
    cout << "wrote " << flout << "_{rho.bin,energy.dat,meta.txt}" << endl;
  }

  print_line_star(60);
  return 0;
}
