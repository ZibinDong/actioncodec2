// Continuous-segment preprocessing kernels. Policy and statistics stay in Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <vector>
#include "quaternion.hpp"

namespace py = pybind11;
namespace {
using D64=py::array_t<double,py::array::c_style|py::array::forcecast>;
using F32=py::array_t<float,py::array::c_style|py::array::forcecast>;

double step_norm(const double* a,const double* b,int64_t width,bool single) {
  if(single) {
    float sum=0;
    for(int64_t d=0;d<width;++d) {const float v=static_cast<float>(b[d])-static_cast<float>(a[d]);sum+=v*v;}
    return std::sqrt(sum);
  }
  double sum=0;
  for(int64_t d=0;d<width;++d){double v=b[d]-a[d];sum+=v*v;}
  return std::sqrt(sum);
}

D64 step_norms(const D64& values,bool single) {
  if(values.ndim()!=2)throw std::invalid_argument("expected [T,D] values");
  const int64_t steps=std::max<int64_t>(0,values.shape(0)-1),width=values.shape(1);
  D64 output(steps);
  py::gil_scoped_release release;
  for(int64_t t=0;t<steps;++t)output.mutable_data()[t]=step_norm(values.data()+t*width,values.data()+(t+1)*width,width,single);
  return output;
}

std::pair<int64_t,int64_t> boundary_trim(const D64& values,double tau,int64_t idle,
                                        int64_t context,bool legacy,bool single) {
  if(values.ndim()!=2 || context<1)throw std::invalid_argument("invalid trimming input");
  int64_t length=values.shape(0),width=values.shape(1);
  if(length<2)return {0,length};
  py::gil_scoped_release release;
  int64_t first=-1,last=-1;
  for(int64_t t=0;t<length-1;++t) {
    if(step_norm(values.data()+t*width,values.data()+(t+1)*width,width,single)>tau) {
      if(first<0)first=t;
      last=t;
    }
  }
  if(first<0)return legacy?std::make_pair(length,length):std::make_pair(int64_t{0},length);
  const int64_t trailing=length-2-last;
  int64_t start=first>=idle?(legacy?first+1:std::max<int64_t>(0,first-context+1)):0;
  int64_t stop=trailing>=idle?last+2:length;
  return {start,legacy?std::max(start,stop):std::max(start+1,stop)};
}

py::dict trim_deviation(const D64& values,int64_t start,int64_t stop,
                        const std::string& profile,int64_t gripper) {
  if(values.ndim()!=2 || start<0 || start>=stop || stop>values.shape(0))
    throw std::invalid_argument("trim slice must retain at least one row");
  const int64_t width=values.shape(1),length=values.shape(0);
  std::vector<int64_t> motion;
  for(int64_t d=0;d<width;++d)if(d!=gripper)motion.push_back(d);
  if(profile=="eef" && motion.size()<6)throw std::invalid_argument("EEF needs six motion columns");
  std::vector<double> maximum(motion.size(),0);
  double translation=0,rotation=0;bool changed=false;
  {
    py::gil_scoped_release release;
    for(int side=0;side<2;++side) {
      int64_t begin=side?stop:0,end=side?length:start,anchor=side?stop-1:start;
      const double* a=values.data()+anchor*width;
      for(int64_t t=begin;t<end;++t) {
        const double* v=values.data()+t*width;
        for(size_t d=0;d<motion.size();++d)maximum[d]=std::max(maximum[d],std::abs(v[motion[d]]-a[motion[d]]));
        if(gripper>=0)changed=changed || v[gripper]!=a[gripper];
        if(profile=="eef") {
          double distance=0,ra[3],rv[3],relative[3];
          for(int d=0;d<3;++d) {
            double delta=v[motion[d]]-a[motion[d]];distance+=delta*delta;
            ra[d]=a[motion[d+3]];rv[d]=v[motion[d+3]];
          }
          translation=std::max(translation,std::sqrt(distance));
          ac2q::to_rotvec(ac2q::multiply(ac2q::conjugate(ac2q::from_rotvec(ra)),ac2q::from_rotvec(rv)),relative);
          rotation=std::max(rotation,std::sqrt(relative[0]*relative[0]+relative[1]*relative[1]+relative[2]*relative[2]));
        }
      }
    }
  }
  py::dict result;
  result["motion_max_error"]=maximum;result["translation_max_error_m"]=translation;
  result["rotation_max_error_rad"]=rotation;result["gripper_changed"]=changed;
  return result;
}

// Small normalized Vandermonde least-squares systems, solved in extended precision.
std::vector<double> polynomial_weights(int window,int degree) {
  const int p=degree+1;
  std::vector<long double> basis(window*p),matrix(p*2*p,0);
  for(int i=0;i<window;++i) {
    long double x=2.L*i/(window-1)-1.L,v=1;
    for(int d=0;d<p;++d){basis[i*p+d]=v;v*=x;}
  }
  for(int a=0;a<p;++a) {
    for(int b=0;b<p;++b)for(int i=0;i<window;++i)matrix[a*2*p+b]+=basis[i*p+a]*basis[i*p+b];
    matrix[a*2*p+p+a]=1;
  }
  for(int a=0;a<p;++a) {
    int pivot=a;
    for(int b=a+1;b<p;++b)if(std::abs(matrix[b*2*p+a])>std::abs(matrix[pivot*2*p+a]))pivot=b;
    if(std::abs(matrix[pivot*2*p+a])<1e-25L)throw std::invalid_argument("singular polynomial fit");
    for(int j=0;j<2*p;++j)std::swap(matrix[a*2*p+j],matrix[pivot*2*p+j]);
    long double scale=matrix[a*2*p+a];
    for(int j=0;j<2*p;++j)matrix[a*2*p+j]/=scale;
    for(int b=0;b<p;++b)if(b!=a) {
      const long double factor=matrix[b*2*p+a];
      for(int j=0;j<2*p;++j)matrix[b*2*p+j]-=factor*matrix[a*2*p+j];
    }
  }
  std::vector<double> weights(window*window);
  for(int target=0;target<window;++target)for(int i=0;i<window;++i) {
    long double w=0;
    for(int a=0;a<p;++a)for(int b=0;b<p;++b)w+=basis[target*p+a]*matrix[a*2*p+p+b]*basis[i*p+b];
    weights[target*window+i]=static_cast<double>(w);
  }
  return weights;
}

F32 savgol(const D64& values,int window,int degree) {
  if(values.ndim()!=2 || window<3 || window%2!=1 || degree<0 || degree>=window || window>values.shape(0))
    throw std::invalid_argument("invalid Savitzky-Golay shape or polynomial order");
  std::vector<double> weights;
  {
    static std::mutex mutex;
    static std::map<std::pair<int,int>,std::vector<double>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto key=std::make_pair(window,degree);auto it=cache.find(key);
    if(it==cache.end())it=cache.emplace(key,polynomial_weights(window,degree)).first;
    weights=it->second;
  }
  const int64_t length=values.shape(0),width=values.shape(1);
  F32 output({length,width});
  py::gil_scoped_release release;
  for(int64_t t=0;t<length;++t) {
    int64_t begin=std::clamp<int64_t>(t-window/2,0,length-window),target=t-begin;
    for(int64_t d=0;d<width;++d) {
      double value=0;
      for(int i=0;i<window;++i)value+=weights[target*window+i]*values.data()[(begin+i)*width+d];
      output.mutable_data()[t*width+d]=static_cast<float>(value);
    }
  }
  return output;
}

D64 median(const D64& values,int window,bool nearest) {
  if(values.ndim()!=1 || window<1 || window%2!=1)throw std::invalid_argument("invalid median filter input");
  const int64_t length=values.size();D64 output(length);
  py::gil_scoped_release release;
  std::vector<double> sample(window);
  for(int64_t i=0;i<length;++i) {
    for(int j=0;j<window;++j) {
      int64_t k=i+j-window/2;
      sample[j]=(k<0 || k>=length)?(nearest?values.data()[std::clamp<int64_t>(k,0,length-1)]:0):values.data()[k];
    }
    std::nth_element(sample.begin(),sample.begin()+window/2,sample.end());
    output.mutable_data()[i]=sample[window/2];
  }
  return output;
}

F32 ease(const D64& values,const D64& limits) {
  if(values.ndim()!=2 || limits.ndim()!=1 || limits.size()!=values.shape(1))
    throw std::invalid_argument("invalid easing shape");
  int64_t length=values.shape(0),width=values.shape(1);F32 output({length,width});
  py::gil_scoped_release release;
  for(int64_t d=0;d<width;++d) {
    if(length==0)continue;
    double debt=0,sum=0,limit=limits.data()[d];
    output.mutable_data()[d]=static_cast<float>(values.data()[d]);
    for(int64_t t=1;t<length;++t) {
      double step=values.data()[t*width+d]-values.data()[(t-1)*width+d],candidate=step+debt;
      if(std::abs(candidate)<=limit){step=candidate;debt=0;}
      else {step=(candidate>0?1:-1)*limit;debt=candidate-step;}
      sum+=step;output.mutable_data()[t*width+d]=static_cast<float>(values.data()[d]+sum);
    }
  }
  return output;
}

F32 gripper(const D64& values,const std::string& operation,double first,double second) {
  if(values.ndim()!=2 || values.shape(1)!=1)throw std::invalid_argument("gripper action must be [T,1]");
  F32 output({values.shape(0),int64_t{1}});
  py::gil_scoped_release release;
  for(int64_t i=0;i<values.size();++i) {
    float v=static_cast<float>(values.data()[i]);
    if(!std::isfinite(v))throw std::invalid_argument("gripper fraction must be finite");
    if(operation=="normalize") {
      const float span=static_cast<float>(second-first);
      output.mutable_data()[i]=std::clamp((v-static_cast<float>(first))/span,0.F,1.F);
    } else if(operation=="binary")output.mutable_data()[i]=v>=static_cast<float>(first)?1.F:0.F;
    else if(operation=="eleven")output.mutable_data()[i]=std::nearbyint(v*10.F)/10.F;
    else throw std::invalid_argument("invalid gripper operation");
  }
  return output;
}
std::vector<double> max_change(const D64& values,const D64& reference,
                               const std::vector<int64_t>& columns,bool single) {
  if(values.ndim()!=2 || reference.ndim()!=2 || values.shape(0)!=reference.shape(0) || values.shape(1)!=reference.shape(1))
    throw std::invalid_argument("mismatched change metric shapes");
  std::vector<double> result(columns.size(),0);const int64_t width=values.shape(1);
  py::gil_scoped_release release;
  for(size_t i=0;i<columns.size();++i) {
    const int64_t d=(columns[i]%width+width)%width;
    for(int64_t t=0;t<values.shape(0);++t) {
      double change=single?std::abs(static_cast<float>(values.data()[t*width+d])-static_cast<float>(reference.data()[t*width+d]))
                          :std::abs(values.data()[t*width+d]-reference.data()[t*width+d]);
      result[i]=std::max(result[i],change);
    }
  }
  return result;
}

int64_t overspeed_steps(const D64& values,const D64& limits,bool single) {
  if(values.ndim()!=2 || limits.size()!=values.shape(1))throw std::invalid_argument("invalid overspeed metric shapes");
  int64_t count=0,width=values.shape(1);py::gil_scoped_release release;
  for(int64_t t=1;t<values.shape(0);++t)for(int64_t d=0;d<width;++d) {
    double delta=single?static_cast<float>(values.data()[t*width+d])-static_cast<float>(values.data()[(t-1)*width+d])
                       :values.data()[t*width+d]-values.data()[(t-1)*width+d];
    if(std::abs(delta)>limits.data()[d]){++count;break;}
  }
  return count;
}

F32 wrap_angles(const D64& values) {
  std::vector<py::ssize_t> shape(values.shape(),values.shape()+values.ndim());F32 output(shape);
  const double pi=std::acos(-1.);py::gil_scoped_release release;
  for(py::ssize_t i=0;i<values.size();++i) {
    double remainder=std::fmod(pi-values.data()[i],2*pi);if(remainder<0)remainder+=2*pi;
    output.mutable_data()[i]=static_cast<float>(pi-remainder);
  }
  return output;
}

template<class T>
void unwrap_column(const double* values,double* output,int64_t length,int64_t width,int64_t d) {
  if(!length)return;
  const T pi=static_cast<T>(std::acos(-1.)),period=static_cast<T>(2*std::acos(-1.));
  T sum=0;output[d]=values[d];
  for(int64_t t=1;t<length;++t) {
    const T delta=static_cast<T>(values[t*width+d])-static_cast<T>(values[(t-1)*width+d]);
    T reduced=std::fmod(static_cast<T>(delta+pi),period);if(reduced<0)reduced+=period;
    reduced-=pi;if(reduced==-pi && delta>0)reduced=pi;
    const T correction=std::abs(delta)<pi?0:reduced-delta;sum+=correction;
    output[t*width+d]=static_cast<T>(static_cast<T>(values[t*width+d])+sum);
  }
}
D64 unwrap_angles(const D64& values,bool single) {
  if(values.ndim()!=2)throw std::invalid_argument("unwrap expects [T,D]");
  int64_t length=values.shape(0),width=values.shape(1);D64 output({length,width});
  py::gil_scoped_release release;
  for(int64_t d=0;d<width;++d) {
    if(single)unwrap_column<float>(values.data(),output.mutable_data(),length,width,d);
    else unwrap_column<double>(values.data(),output.mutable_data(),length,width,d);
  }
  return output;
}
}  // namespace

void register_preprocessing(py::module_& module) {
  module.def("step_norms",&step_norms);
  module.def("boundary_trim",&boundary_trim);
  module.def("trim_deviation",&trim_deviation);
  module.def("savgol_smooth",&savgol);
  module.def("median_filter",&median);
  module.def("ease_overspeed",&ease);
  module.def("gripper_transform",&gripper);
  module.def("max_change",&max_change);
  module.def("overspeed_steps",&overspeed_steps);
  module.def("wrap_angles",&wrap_angles);
  module.def("unwrap_angles",&unwrap_angles);
}
