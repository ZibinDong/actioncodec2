#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ac2q {
using Q = std::array<double, 4>;
inline Q normalize(Q q) {
  double n = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  if (n < 1e-12) throw std::invalid_argument("a pose contains a zero quaternion");
  for (double& v : q) v /= n;
  return q;
}
inline Q multiply(const Q& a, const Q& b) {
  return normalize({a[3]*b[0]+b[3]*a[0]+(a[1]*b[2]-a[2]*b[1]),
                    a[3]*b[1]+b[3]*a[1]+(a[2]*b[0]-a[0]*b[2]),
                    a[3]*b[2]+b[3]*a[2]+(a[0]*b[1]-a[1]*b[0]),
                    a[3]*b[3]-(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])});
}
inline Q conjugate(Q q) { for (int i=0;i<3;++i) q[i] = -q[i]; return q; }
inline Q from_rotvec(const double* v) {
  const double a = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  const double s = a >= 1e-12 ? std::sin(a/2)/a : .5;
  return normalize({v[0]*s,v[1]*s,v[2]*s,std::cos(a/2)});
}
inline void to_rotvec(Q q, double* out) {
  q=normalize(q); if(q[3]<0) for(double& x:q)x=-x;
  const double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
  const double a=2*std::atan2(n,std::clamp(q[3],0.,1.));
  const double s=n>=1e-12?a/n:2.;
  for(int j=0;j<3;++j)out[j]=q[j]*s;
}
inline Q from_euler(const double* v) {
  const double sx=std::sin(v[0]/2),sy=std::sin(v[1]/2),sz=std::sin(v[2]/2);
  const double cx=std::cos(v[0]/2),cy=std::cos(v[1]/2),cz=std::cos(v[2]/2);
  return normalize({sx*cy*cz-cx*sy*sz,cx*sy*cz+sx*cy*sz,
                    cx*cy*sz-sx*sy*cz,cx*cy*cz+sx*sy*sz});
}
inline void to_euler(Q q, double* out) {
  q=normalize(q);const auto[x,y,z,w]=q;
  out[0]=std::atan2(2*(w*x+y*z),1-2*(x*x+y*y));
  out[1]=std::asin(std::clamp(2*(w*y-z*x),-1.,1.));
  out[2]=std::atan2(2*(w*z+x*y),1-2*(y*y+z*z));
}
}  // namespace ac2q
