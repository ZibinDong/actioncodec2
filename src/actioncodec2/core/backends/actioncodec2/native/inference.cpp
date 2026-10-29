// Lossless plain-BPE inference and additive physical reconstruction.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>
#include "parallel.hpp"
#include "quaternion.hpp"

namespace py = pybind11;
namespace {
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;



F64 quaternion_map(const F64& values, const std::string& operation) {
  const bool vector_in = operation == "from_rotvec" || operation == "from_euler";
  const int input_width=vector_in?3:4;
  const bool vector_out=operation == "to_rotvec" || operation == "to_euler";
  const int width=vector_out?3:4;
  if(values.ndim()!=2 || values.shape(1)!=input_width)
    throw std::invalid_argument("invalid quaternion operation shape");
  F64 output({values.shape(0),static_cast<int64_t>(width)});
  py::gil_scoped_release release;
  for(int64_t i=0;i<values.shape(0);++i) {
    const double* v=values.data()+i*input_width;
    ac2q::Q q=vector_in?(operation=="from_rotvec"?ac2q::from_rotvec(v):ac2q::from_euler(v))
                         :ac2q::Q{v[0],v[1],v[2],v[3]};
    double* out=output.mutable_data()+i*width;
    if(operation=="to_rotvec")ac2q::to_rotvec(q,out);
    else if(operation=="to_euler")ac2q::to_euler(q,out);
    else {
      if(operation=="normalize")q=ac2q::normalize(q);
      else if(operation=="conjugate")q=ac2q::conjugate(q);
      else if(!vector_in)throw std::invalid_argument("unknown quaternion operation");
      std::copy(q.begin(),q.end(),out);
    }
  }
  return output;
}

F64 quaternion_multiply(const F64& first, const F64& second) {
  if(first.ndim()!=2 || first.shape(1)!=4 || second.ndim()!=2 ||
     second.shape(1)!=4 || first.shape(0)!=second.shape(0))
    throw std::invalid_argument("invalid quaternion product shape");
  F64 output({first.shape(0),int64_t{4}});
  py::gil_scoped_release release;
  for(int64_t i=0;i<first.shape(0);++i) {
    auto* a=first.data()+4*i;auto* b=second.data()+4*i;
    auto q=ac2q::multiply({a[0],a[1],a[2],a[3]},{b[0],b[1],b[2],b[3]});
    std::copy(q.begin(),q.end(),output.mutable_data()+4*i);
  }
  return output;
}

F32 so3_reconstruct(const F64& table, const I64& cells, const F32& previous,
                    const F32& scale, int input_order, int primitive_order,
                    bool left, bool rotation_vector, double radius) {
  if(cells.ndim()!=3 || cells.shape(2)!=3 || table.ndim()!=2 || table.shape(0)!=3 ||
     previous.ndim()!=2 || previous.shape(0)!=cells.shape(0) || previous.shape(1)!=3 || scale.size()!=3)
    throw std::invalid_argument("invalid SO(3) reconstruction shapes");
  const int64_t batch=cells.shape(0),steps=cells.shape(1);
  F32 output({batch,steps,int64_t{3}});
  py::gil_scoped_release release;
  ac2::parallel_for(batch,0,[&](int64_t b) {
    double boundary[3];
    for(int d=0;d<3;++d)boundary[d]=static_cast<double>(previous.data()[3*b+d])*scale.data()[d];
    auto increment=rotation_vector?ac2q::from_rotvec(boundary):ac2q::from_euler(boundary);
    ac2q::Q state{0,0,0,1};
    float accumulated[3]={0,0,0};
    auto apply=[&](ac2q::Q a,ac2q::Q v){return left?ac2q::multiply(v,a):ac2q::multiply(a,v);};
    for(int64_t t=0;t<steps;++t) {
      double v[3],represented[3];
      for(int d=0;d<3;++d) {
        const int64_t cell=cells.data()[(b*steps+t)*3+d];
        if(cell<0 || cell>=table.shape(1))throw std::invalid_argument("invalid primitive cell");
        v[d]=table.data()[d*table.shape(1)+cell];
      }
      if(primitive_order==2) {
        const double norm=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        const double factor=std::min(1.,radius/std::max(norm,std::numeric_limits<double>::min()));
        for(double& x:v)x*=factor;
        increment=apply(increment,ac2q::from_rotvec(v));
        state=apply(state,increment);
        auto q=input_order==1?increment:state;
        if(rotation_vector)ac2q::to_rotvec(q,represented);else ac2q::to_euler(q,represented);
      } else if(input_order==1) {
        if(rotation_vector)std::copy(v,v+3,represented);
        else {for(double& x:v)x=static_cast<float>(x);ac2q::to_euler(ac2q::from_rotvec(v),represented);}
      } else {
        state=apply(state,ac2q::from_rotvec(v));
        if(rotation_vector)ac2q::to_rotvec(state,represented);else ac2q::to_euler(state,represented);
      }
      for(int d=0;d<3;++d) {
        float value=static_cast<float>(represented[d]/scale.data()[d]);
        if(primitive_order==1 && input_order==1) {
          const float next=accumulated[d]+value;value=next-accumulated[d];accumulated[d]=next;
        }
        output.mutable_data()[(b*steps+t)*3+d]=value;
      }
    }
  });
  return output;
}

F32 additive_reconstruct(const F64& table, const I64& cells, const F32& previous,
                         const F32& scale, int input_order, int primitive_order) {
  if (cells.ndim() != 3 || table.ndim() != 2 || scale.ndim() != 1 ||
      cells.shape(2) != table.shape(0) || scale.size() != cells.shape(2) ||
      previous.ndim() != 2 || previous.shape(0) != cells.shape(0) || previous.shape(1) != cells.shape(2))
    throw std::invalid_argument("invalid additive reconstruction shapes");
  if (primitive_order != 2 || input_order < 0 || input_order > 2)
    throw std::invalid_argument("invalid differential order");
  const int64_t batch = cells.shape(0), steps = cells.shape(1), width = cells.shape(2);
  F32 output({batch, steps, width});
  py::gil_scoped_release release;
  ac2::parallel_for(batch, 0, [&](int64_t b) {
    for (int64_t d = 0; d < width; ++d) {
      double change_sum = 0, position = 0;
      const float boundary = previous.data()[b * width + d] * scale.data()[d];
      for (int64_t t = 0; t < steps; ++t) {
        const int64_t index = (b * steps + t) * width + d, cell = cells.data()[index];
        if (cell < 0 || cell >= table.shape(1)) throw std::invalid_argument("invalid cell");
        const double primitive = table.data()[d * table.shape(1) + cell];
        change_sum += primitive;
        const double velocity = change_sum + boundary;
        position += velocity;
        const double value = input_order == 2 ? primitive : input_order == 1 ? velocity : position;
        output.mutable_data()[index] = static_cast<float>(value / scale.data()[d]);
      }
    }
  });
  return output;
}

I64 choose_cells(const F32& table,const I64& bins,const F32& request,
                  const F32& direction,const F32& limits,bool smooth) {
  if(table.ndim()!=2 || request.ndim()!=2 || direction.ndim()!=2 ||
     direction.shape(0)!=request.shape(0) || direction.shape(1)!=request.shape(1) ||
     table.shape(0)!=request.shape(1) || bins.size()!=request.shape(1) || limits.size()!=request.shape(1))
    throw std::invalid_argument("invalid primitive selection shapes");
  I64 output({request.shape(0),request.shape(1)});
  py::gil_scoped_release release;
  const int64_t width=request.shape(1);
  for(int64_t i=0;i<request.shape(0);++i)for(int64_t d=0;d<width;++d) {
    int64_t nearest=0,feasible=-1,preferred=-1;
    float best=INFINITY,best_feasible=INFINITY,best_preferred=INFINITY;
    const float q=request.data()[i*width+d],sign=direction.data()[i*width+d];
    for(int64_t j=0;j<bins.data()[d];++j) {
      float p=table.data()[d*table.shape(1)+j],distance=std::abs(q-p);
      if(distance<best){best=distance;nearest=j;}
      if(distance<=limits.data()[d]+8*std::numeric_limits<float>::epsilon()) {
        if(distance<best_feasible){best_feasible=distance;feasible=j;}
        if((p>0?1.F:p<0?-1.F:0.F)*sign>=0 && distance<best_preferred) {
          best_preferred=distance;preferred=j;
        }
      }
    }
    output.mutable_data()[i*width+d]=smooth?(preferred>=0?preferred:feasible>=0?feasible:nearest):nearest;
  }
  return output;
}

F32 additive_decode(const F32& table,const I64& cells,const F32& scale,
                     const F32& lower,const F32& upper,int input_order,int primitive_order) {
  if(cells.ndim()!=3 || table.ndim()!=2 || table.shape(0)!=cells.shape(2) ||
     scale.size()!=cells.shape(2) || lower.size()!=cells.shape(2) || upper.size()!=cells.shape(2) ||
     primitive_order<0 || primitive_order>1 || input_order<0 || input_order>2)
    throw std::invalid_argument("invalid additive decoding metadata");
  const int64_t batch=cells.shape(0),steps=cells.shape(1),width=cells.shape(2);
  F32 output({batch,steps,width});
  py::gil_scoped_release release;
  ac2::parallel_for(batch,0,[&](int64_t b) {
    for(int64_t d=0;d<width;++d) {
      float position=0;
      for(int64_t t=0;t<steps;++t) {
        const int64_t index=(b*steps+t)*width+d,cell=cells.data()[index];
        if(cell<0 || cell>=table.shape(1))throw std::invalid_argument("invalid primitive cell");
        const float picked=table.data()[d*table.shape(1)+cell];
        position=primitive_order?std::clamp(position+picked,lower.data()[d],upper.data()[d]):picked;
        output.mutable_data()[index]=position;
      }
      for(int order=0;order<input_order;++order)for(int64_t t=steps-1;t>0;--t)
        output.mutable_data()[(b*steps+t)*width+d]-=output.data()[(b*steps+t-1)*width+d];
      for(int64_t t=0;t<steps;++t)output.mutable_data()[(b*steps+t)*width+d]/=scale.data()[d];
    }
  });
  return output;
}

I64 spatial_expand(const std::vector<int64_t>& tokens,const I64& expansions,
                    const I64& supports,uint64_t full) {
  if(expansions.ndim()!=2 || supports.ndim()!=1 || expansions.shape(0)!=supports.size())
    throw std::invalid_argument("invalid spatial metadata");
  const int64_t width=expansions.shape(1);I64 output(width);
  std::fill(output.mutable_data(),output.mutable_data()+width,-1);
  uint64_t covered=0;
  py::gil_scoped_release release;
  for(int64_t token:tokens) {
    if(token<0 || token>=supports.size())throw std::invalid_argument("token is outside the vocabulary");
    const uint64_t support=supports.data()[token];
    if(covered & support)throw std::invalid_argument("token supports overlap");
    covered|=support;
    for(int64_t d=0;d<width;++d) {
      const int64_t value=expansions.data()[token*width+d];
      if(value>=0)output.mutable_data()[d]=value;
    }
  }
  if(covered!=full)throw std::invalid_argument("token set does not exactly cover the requested active dimensions");
  return output;
}
F32 additive_transform(const F32& values,int order,bool integrate) {
  if(values.ndim()!=3 || order<0 || order>2)throw std::invalid_argument("expected [B,T,D] and order 0..2");
  const int64_t batch=values.shape(0),steps=values.shape(1),width=values.shape(2);
  F32 output({batch,steps,width});std::copy(values.data(),values.data()+values.size(),output.mutable_data());
  py::gil_scoped_release release;
  for(int level=0;level<order;++level)for(int64_t b=0;b<batch;++b)for(int64_t d=0;d<width;++d) {
    if(integrate) {
      float sum=0;
      for(int64_t t=0;t<steps;++t) {const int64_t i=(b*steps+t)*width+d;sum+=output.data()[i];output.mutable_data()[i]=sum;}
    } else {
      for(int64_t t=steps-1;t>0;--t)output.mutable_data()[(b*steps+t)*width+d]-=output.data()[(b*steps+t-1)*width+d];
    }
  }
  return output;
}
}  // namespace

void register_inference(py::module_& module) {
  module.def("additive_reconstruct", &additive_reconstruct);
  module.def("so3_reconstruct", &so3_reconstruct);
  module.def("quaternion_map", &quaternion_map);
  module.def("quaternion_multiply", &quaternion_multiply);
  module.def("choose_cells", &choose_cells);
  module.def("additive_decode", &additive_decode);
  module.def("spatial_expand", &spatial_expand);
  module.def("additive_transform", &additive_transform);
}
