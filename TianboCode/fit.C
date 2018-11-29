#include <iostream>
#include <fstream>
#include <cmath>

#include "Math/GSLIntegrator.h"
#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/Functor.h"
#include "Math/Factory.h"
#include "Math/WrappedTF1.h"
#include "Math/WrappedParamFunction.h"
#include "TF1.h"
#include "Math/Minimizer.h"
#include "TString.h"
#include "TMatrixDEigen.h"
#include "TMath.h"


using namespace std;

double Beta(double a, double b){
  return TMath::Beta(a, b);
}
  

double Variables[20], Values[20], Errors[20];
int LoadData(){
  ifstream file("data.txt");
  //file.ignore(300, '\n');
  int i = 0;
  while ( file >> Variables[i] >> Values[i] >> Errors[i]){
    cout << Variables[i] << endl;
    i++;
  }
  file.close();
  return 0;
}

double integrand(const double * xp, const double * par){
  //par: a, b, c, d, w
  double x = xp[0];
  double w = par[4];
  return cos(x * w) * pow(x, par[0]) * pow(1.0 - x, par[1]) * (1.0 + par[2] * x + par[3] * sqrt(x)) / (Beta(par[0] + 1.0, par[1] + 1.0) + par[2] * Beta(par[0] + 2.0, par[1] + 1.0) + par[3] * Beta(par[0] + 1.5, par[1] + 1.0));
}

double lcs(const double w, const double * par){
  double para[5] = {par[0], par[1], par[2], par[3], w};
  TF1 fs("integrand", &integrand, 0.0, 1.0, 5);
  ROOT::Math::WrappedTF1 wfs(fs);
  wfs.SetParameters(para);
  ROOT::Math::IntegratorOneDim ig(ROOT::Math::IntegrationOneDim::kADAPTIVE, 0.0, 1e-5, 100000);
  ig.SetFunction(wfs);
  return 2.0 / pow(M_PI, 2) * ig.Integral(0.0, 1.0);
}
  
double Chi2(const double * par){
  double sum = 0.0;
  for (int i = 0; i < 16; i++){
    sum += pow(lcs(Variables[i], par) - Values[i], 2) / pow(Errors[i], 2);
  }
  return sum;
}

TMatrixD Minimize(const double * init, double * cent){
  ROOT::Math::Minimizer * min = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad");
  min->SetMaxFunctionCalls(10000000);
  min->SetTolerance(1e-2);
  min->SetPrintLevel(0);
  ROOT::Math::Functor f(&Chi2, 4);
  min->SetFunction(f);
  min->SetLimitedVariable(0, "a", init[0], 1e-4, -0.7, -0.2);
  min->SetLimitedVariable(1, "b", init[1], 1e-4, 1.0, 2.5);
  min->SetVariable(2, "c", init[2], 1e-4);
  min->SetVariable(3, "d", init[3], 1e-4);
  min->Minimize();
  min->PrintResults();
  double cov[16];
  min->GetCovMatrix(cov);
  TMatrixD Cov(4, 4);
  for (int i = 0; i < 4; i++){
    for (int j = 0; j < 4; j++){
      Cov(i, j) = cov[i*4 + j];
    }
  }
  const double * xs = min->X();
  for (int i = 0; i < 4; i++)
    cent[i] = xs[i];
  Cov.Print();
  return Cov;
}

int Calculate(TMatrixD Cov, double * cent){
  TMatrixDEigen eigen(Cov);
  TMatrixD val = eigen.GetEigenValues();
  TMatrixD vec = eigen.GetEigenVectors();
  double par[4] = {cent[0], cent[1], cent[2], cent[3]};
  double factor = 0.1;
  double par1p[4] = {cent[0] + factor * sqrt(val(0,0)) * vec(0,0),
		     cent[1] + factor * sqrt(val(0,0)) * vec(1,0),
		     cent[2] + factor * sqrt(val(0,0)) * vec(2,0),
		     cent[3] + factor * sqrt(val(0,0)) * vec(3,0)};
  double par1m[4] = {cent[0] - factor * sqrt(val(0,0)) * vec(0,0),
		     cent[1] - factor * sqrt(val(0,0)) * vec(1,0),
		     cent[2] - factor * sqrt(val(0,0)) * vec(2,0),
		     cent[3] - factor * sqrt(val(0,0)) * vec(3,0)};
  double par2p[4] = {cent[0] + factor * sqrt(val(1,1)) * vec(0,1),
		     cent[1] + factor * sqrt(val(1,1)) * vec(1,1),
		     cent[2] + factor * sqrt(val(1,1)) * vec(2,1),
		     cent[3] + factor * sqrt(val(1,1)) * vec(3,1)};
  double par2m[4] = {cent[0] - factor * sqrt(val(1,1)) * vec(0,1),
		     cent[1] - factor * sqrt(val(1,1)) * vec(1,1),
		     cent[2] - factor * sqrt(val(1,1)) * vec(2,1),
		     cent[3] - factor * sqrt(val(1,1)) * vec(3,1)};
  double par3p[4] = {cent[0] + factor * sqrt(val(2,2)) * vec(0,2),
		     cent[1] + factor * sqrt(val(2,2)) * vec(1,2),
		     cent[2] + factor * sqrt(val(2,2)) * vec(2,2),
		     cent[3] + factor * sqrt(val(2,2)) * vec(3,2)};
  double par3m[4] = {cent[0] - factor * sqrt(val(2,2)) * vec(0,2),
		     cent[1] - factor * sqrt(val(2,2)) * vec(1,2),
		     cent[2] - factor * sqrt(val(2,2)) * vec(2,2),
		     cent[3] - factor * sqrt(val(2,2)) * vec(3,2)};
  double par4p[4] = {cent[0] + factor * sqrt(val(3,3)) * vec(0,3),
		     cent[1] + factor * sqrt(val(3,3)) * vec(1,3),
		     cent[2] + factor * sqrt(val(3,3)) * vec(2,3),
		     cent[3] + factor * sqrt(val(3,3)) * vec(3,3)};
  double par4m[4] = {cent[0] - factor * sqrt(val(3,3)) * vec(0,3),
		     cent[1] - factor * sqrt(val(3,3)) * vec(1,3),
		     cent[2] - factor * sqrt(val(3,3)) * vec(2,3),
		     cent[3] - factor * sqrt(val(3,3)) * vec(3,3)};
  double w, F, dF;
  FILE * file = fopen("band.txt", "w");
  //fprintf(file, "w\tF\tdF\n");
  for (int i = 0; i < 1000; i++){
    w = i * 0.01;
    F = lcs(w, par);
    dF = sqrt(pow(lcs(w, par1p) - lcs(w, par1m), 2) + pow(lcs(w, par2p) - lcs(w, par2m), 2) + pow(lcs(w, par3p) - lcs(w, par3m), 2) + pow(lcs(w, par4p) - lcs(w, par4m), 2)) / 2.0 / factor;
    fprintf(file, "%.6E\t%.6E\t%.6E\n", w, F, dF);
  }
  fclose(file);

  cout << "0:\t" << par[0] << "\t" << par[1] << "\t" << par[2] << "\t" << par[3] << endl;
  cout << "1+:\t" << par1p[0] << "\t" << par1p[1] << "\t" << par1p[2] << "\t" << par1p[3] << endl;
  cout << "1-:\t" << par1m[0] << "\t" << par1m[1] << "\t" << par1m[2] << "\t" << par1m[3] << endl;
  cout << "2+:\t" << par2p[0] << "\t" << par2p[1] << "\t" << par2p[2] << "\t" << par2p[3] << endl;
  cout << "2-:\t" << par2m[0] << "\t" << par2m[1] << "\t" << par2m[2] << "\t" << par2m[3] << endl;
  cout << "3+:\t" << par3p[0] << "\t" << par3p[1] << "\t" << par3p[2] << "\t" << par3p[3] << endl;
  cout << "3-:\t" << par3m[0] << "\t" << par3m[1] << "\t" << par3m[2] << "\t" << par3m[3] << endl;
  cout << "4+:\t" << par4p[0] << "\t" << par4p[1] << "\t" << par4p[2] << "\t" << par4p[3] << endl;
  cout << "4-:\t" << par4m[0] << "\t" << par4m[1] << "\t" << par4m[2] << "\t" << par4m[3] << endl;

  
  return 0;
}

int main(const int argc, const char * argv[]){

  LoadData();

  double par[4] = {-0.5, 2.0, -4.2, 11.0};
  double cent[4];
  TMatrixD Cov = Minimize(par, cent);

  Calculate(Cov, cent);

  return 0;
}
