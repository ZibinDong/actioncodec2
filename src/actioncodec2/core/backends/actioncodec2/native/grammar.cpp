// Coverage grammars and multi-component routing, including time-major streams.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include "parallel.hpp"
#include "token_streams.hpp"

namespace py=pybind11;
namespace {
using State=std::vector<int64_t>;
using States=std::vector<State>;
class Grammar {
 public:
  int64_t horizon,vocabulary;
  virtual ~Grammar()=default;
  virtual State initial()const=0;
  virtual State advance(const State&,int64_t)const=0;
  virtual int64_t covered(const State&)const=0;
  virtual int64_t next_dimension(const State&)const=0;
  virtual bool allowed(const State&,int64_t)const=0;
  virtual std::shared_ptr<Grammar> clone(int64_t)const=0;
  py::array_t<bool> mask(const State& state)const {
    py::array_t<bool> result(vocabulary);
    py::gil_scoped_release release;
    for(int64_t token=0;token<vocabulary;++token)result.mutable_data()[token]=allowed(state,token);
    return result;
  }
};
class SetGrammar:public Grammar {
  uint64_t full_;
  using Patterns=std::vector<std::vector<std::pair<int64_t,uint64_t>>>;
  std::shared_ptr<const std::vector<int64_t>> roots_;
  std::shared_ptr<const Patterns> patterns_;
  std::shared_ptr<const std::vector<bool>> scope_;
 public:
  SetGrammar(int64_t steps,uint64_t full,std::vector<int64_t> roots,
             std::vector<std::vector<std::pair<int64_t,uint64_t>>> patterns)
      :full_(full),roots_(std::make_shared<const std::vector<int64_t>>(std::move(roots))),
       patterns_(std::make_shared<const Patterns>(std::move(patterns))) {
    if(steps<1 || !full || roots_->size()!=patterns_->size())throw std::invalid_argument("invalid grammar metadata");
    horizon=steps;vocabulary=roots_->size();
    std::vector<bool> scopes;
    for(const auto& pattern:*patterns_) {
      bool scope=true;
      for(auto[offset,support]:pattern) {
        if(offset<0)throw std::invalid_argument("negative grammar pattern offset");
        scope=scope && !(support & ~full_);
      }
      scopes.push_back(scope);
    }
    scope_=std::make_shared<const std::vector<bool>>(std::move(scopes));
  }
  std::shared_ptr<Grammar> clone(int64_t steps)const override {
    if(steps<1)throw std::invalid_argument("horizon must be positive");
    auto result=std::make_shared<SetGrammar>(*this);result->horizon=steps;return result;
  }
  State initial()const override{return State(horizon,0);}
  int64_t covered(const State& state)const override {
    if(static_cast<int64_t>(state.size())!=horizon)throw std::invalid_argument("invalid grammar state");
    for(int64_t i=0;i<horizon;++i)if(static_cast<uint64_t>(state[i])!=full_)return i;
    return horizon;
  }
  int64_t next_dimension(const State& state)const override {
    int64_t front=covered(state);if(front==horizon)return -1;
    uint64_t missing=full_ & ~static_cast<uint64_t>(state[front]);
    int64_t d=0;while(!(missing & 1)){missing>>=1;++d;}return d;
  }
  bool allowed(const State& state,int64_t token)const override {
    if(token<0 || token>=vocabulary || !(*scope_)[token])return false;
    const int64_t front=covered(state);
    if(front==horizon || (*roots_)[token]!=next_dimension(state))return false;
    for(auto[offset,support]:(*patterns_)[token]) {
      if(front+offset>=horizon || (static_cast<uint64_t>(state[front+offset]) & support))return false;
    }
    return true;
  }
  State advance(const State& state,int64_t token)const override {
    if(!allowed(state,token))return {};
    State result=state;int64_t front=covered(state);
    for(auto[offset,support]:(*patterns_)[token])result[front+offset]|=static_cast<int64_t>(support);
    return result;
  }
};
class PlainGrammar:public Grammar {
  std::shared_ptr<const std::vector<int64_t>> lengths_,dims_;
  std::shared_ptr<const std::vector<bool>> phase_;
 public:
  PlainGrammar(int64_t steps,std::vector<int64_t> lengths,std::vector<int64_t> dims,
                py::array_t<bool,py::array::c_style|py::array::forcecast> phase)
      :lengths_(std::make_shared<const std::vector<int64_t>>(std::move(lengths))),
       dims_(std::make_shared<const std::vector<int64_t>>(std::move(dims))) {
    horizon=steps;vocabulary=lengths_->size();
    if(steps<1 || dims_->empty() || phase.ndim()!=2 || phase.shape(0)!=static_cast<int64_t>(dims_->size()) || phase.shape(1)!=vocabulary)
      throw std::invalid_argument("invalid plain-BPE grammar metadata");
    for(auto value:*lengths_)if(value<1)throw std::invalid_argument("invalid token expansion");
    phase_=std::make_shared<const std::vector<bool>>(phase.data(),phase.data()+phase.size());
  }
  std::shared_ptr<Grammar> clone(int64_t steps)const override {
    if(steps<1)throw std::invalid_argument("horizon must be positive");
    auto result=std::make_shared<PlainGrammar>(*this);result->horizon=steps;return result;
  }
  State initial()const override{return {0};}
  int64_t covered(const State& state)const override {
    if(state.size()!=1 || state[0]<0)throw std::invalid_argument("invalid grammar state");
    return state[0]/dims_->size();
  }
  int64_t next_dimension(const State& state)const override{return (*dims_)[state[0]%dims_->size()];}
  bool allowed(const State& state,int64_t token)const override {
    return token>=0 && token<vocabulary && state[0]+(*lengths_)[token]<=horizon*static_cast<int64_t>(dims_->size()) && (*phase_)[(state[0]%dims_->size())*vocabulary+token];
  }
  State advance(const State& state,int64_t token)const override {
    if(!allowed(state,token))return {};
    return {state[0]+(*lengths_)[token]};
  }
};
using Grammars=std::vector<std::shared_ptr<Grammar>>;
int64_t next_component(const Grammars& grammars,const States& states,bool time_major) {
  int64_t selected=grammars.size(),minimum=INT64_MAX;
  for(size_t i=0;i<grammars.size();++i) {
    const int64_t front=grammars[i]->covered(states[i]);
    if(front>=grammars[i]->horizon)continue;
    if(!time_major)return i;
    if(front<minimum){minimum=front;selected=i;}
  }
  return selected;
}
States initial(const Grammars& grammars) {
  States states;for(const auto& grammar:grammars)states.push_back(grammar->initial());return states;
}
bool advance(const Grammars& grammars,const std::vector<int64_t>& offsets,States& states,int64_t index,int64_t token) {
  if(index<0 || index>=static_cast<int64_t>(grammars.size()))return false;
  State next=grammars[index]->advance(states[index],token-offsets[index]);
  if(next.empty())return false;
  states[index]=std::move(next);
  return true;
}
py::tuple state_tuple(const State& state) {
  py::tuple result(state.size());for(size_t i=0;i<state.size();++i)result[i]=state[i];return result;
}
py::object route_advance(const Grammars& grammars,const std::vector<int64_t>& offsets,
                         bool time_major,int64_t index,States states,int64_t token) {
  if(states.size()!=grammars.size() || offsets.size()!=grammars.size())throw std::invalid_argument("invalid routed metadata");
  if(!advance(grammars,offsets,states,index,token))return py::none();
  py::tuple local(states.size());for(size_t i=0;i<states.size();++i)local[i]=state_tuple(states[i]);
  return py::make_tuple(next_component(grammars,states,time_major),local);
}
std::vector<int64_t> interleave(const Grammars& grammars,const std::vector<int64_t>& offsets,
                                bool time_major,const std::vector<std::vector<int64_t>>& streams) {
  if(streams.size()!=grammars.size() || offsets.size()!=grammars.size())throw std::invalid_argument("one stream per component is required");
  States states=initial(grammars);std::vector<size_t> cursor(streams.size());std::vector<int64_t> output;
  int64_t index;
  while((index=next_component(grammars,states,time_major))<static_cast<int64_t>(grammars.size())) {
    if(cursor[index]==streams[index].size())throw std::invalid_argument("component stream ends before complete coverage");
    const int64_t token=streams[index][cursor[index]++]+offsets[index];
    if(!advance(grammars,offsets,states,index,token))throw std::invalid_argument("invalid routed component stream");
    output.push_back(token);
  }
  for(size_t i=0;i<cursor.size();++i)if(cursor[i]!=streams[i].size())throw std::invalid_argument("extra component tokens");
  return output;
}
std::vector<std::vector<int64_t>> split(const Grammars& grammars,const std::vector<int64_t>& offsets,
                                       bool time_major,const std::vector<int64_t>& tokens) {
  if(offsets.size()!=grammars.size())throw std::invalid_argument("invalid routed offsets");
  States states=initial(grammars);std::vector<std::vector<int64_t>> output(grammars.size());
  for(int64_t token:tokens) {
    int64_t index=next_component(grammars,states,time_major);
    if(!advance(grammars,offsets,states,index,token))throw std::invalid_argument("invalid routed token coverage");
    output[index].push_back(token-offsets[index]);
  }
  if(next_component(grammars,states,time_major)!=static_cast<int64_t>(grammars.size()))
    throw std::invalid_argument("Incomplete action token stream");
  return output;
}

using Streams=std::vector<std::vector<int64_t>>;
using ComponentStreams=std::vector<Streams>;

Streams interleave_batch(const Grammars& grammars,const std::vector<int64_t>& offsets,
                        bool time_major,const ComponentStreams& streams) {
  if(streams.size()!=grammars.size() || streams.empty() || streams.front().empty())
    throw std::invalid_argument("one non-empty batch per component is required");
  const int64_t batch=streams.front().size();
  for(const auto& rows:streams)if(static_cast<int64_t>(rows.size())!=batch)
    throw std::invalid_argument("component batches must have the same size");
  Streams output(batch);
  py::gil_scoped_release release;
  ac2::parallel_for(batch,0,[&](int64_t b) {
    Streams local;for(const auto& rows:streams)local.push_back(rows[b]);
    output[b]=interleave(grammars,offsets,time_major,local);
  });
  return output;
}

py::tuple decode_batch(const Grammars& prototypes,const std::vector<int64_t>& offsets,
                       bool time_major,const py::iterable& streams,
                       py::array_t<int64_t,py::array::c_style|py::array::forcecast> counts,
                       int64_t cells_per_step) {
  if(prototypes.empty() || counts.ndim()!=1 || cells_per_step<1)
    throw std::invalid_argument("invalid routed batch metadata");
  const auto rows=ac2::checked_token_streams(streams,counts.size(),true);
  int64_t steps=-1;
  ComponentStreams output(prototypes.size(),Streams(rows.size()));
  {
    py::gil_scoped_release release;
    for(const auto& row:rows) {
      int64_t cells=0;
      for(int64_t token:row) {
        if(counts.data()[token]<1)throw std::invalid_argument("token is outside the routed profile vocabularies");
        cells+=counts.data()[token];
      }
      if(cells%cells_per_step || cells/cells_per_step<1)
        throw std::invalid_argument("Incomplete action token stream: " +
            std::to_string(cells_per_step-cells%cells_per_step) +
            " more coordinate values are needed to finish a step. "
            "Constrain generation with codec.grammar(H, fps=...).next_token_mask(prefix) "
            "and check is_complete(tokens) for the intended H before decoding. "
            "Use codec.batch_decode(..., errors='return') to isolate invalid rows; "
            "do not pad truncated tokens with zeros.");
      if(steps>=0 && steps!=cells/cells_per_step)
        throw std::invalid_argument("all token streams must decode to the same length; use codec.batch_decode(...) for independent variable-length outputs");
      steps=cells/cells_per_step;
    }
    Grammars grammars;for(const auto& prototype:prototypes)grammars.push_back(prototype->clone(steps));
    ac2::parallel_for(rows.size(),0,[&](int64_t b) {
      auto local=split(grammars,offsets,time_major,rows[b]);
      for(size_t i=0;i<local.size();++i)output[i][b]=std::move(local[i]);
    });
  }
  return py::make_tuple(steps,output);
}

std::vector<int64_t> complete(const Grammars& grammars,const std::vector<int64_t>& offsets,
                              bool time_major,std::vector<int64_t> tokens,int64_t steps,
                              const std::vector<std::vector<int64_t>>& zero_atoms) {
  if(grammars.empty() || offsets.size()!=grammars.size() || zero_atoms.size()!=grammars.size() ||
     steps<1 || steps>grammars.front()->horizon)throw std::invalid_argument("invalid prefix completion metadata");
  py::gil_scoped_release release;
  States states=initial(grammars);
  for(int64_t token:tokens) {
    int64_t index=next_component(grammars,states,time_major);
    if(!advance(grammars,offsets,states,index,token))throw std::invalid_argument("invalid routed token prefix");
  }
  for(size_t i=0;i<grammars.size();++i)if(grammars[i]->covered(states[i])<steps)
    throw std::invalid_argument("action prefix does not cover the requested frames");
  int64_t index;
  while((index=next_component(grammars,states,time_major))<static_cast<int64_t>(grammars.size())) {
    int64_t dimension=grammars[index]->next_dimension(states[index]);
    if(dimension<0 || dimension>=static_cast<int64_t>(zero_atoms[index].size()))
      throw std::invalid_argument("missing zero-motion primitive atom");
    const int64_t token=offsets[index]+zero_atoms[index][dimension];
    if(!advance(grammars,offsets,states,index,token))throw std::invalid_argument("invalid zero-motion suffix");
    tokens.push_back(token);
  }
  return tokens;
}
}  // namespace

void register_grammar(py::module_& module) {
  py::class_<Grammar,std::shared_ptr<Grammar>>(module,"CoverageGrammar")
      .def("advance",[](const Grammar& g,const State& s,int64_t token)->py::object {
        State next=g.advance(s,token);return next.empty()?py::object(py::none()):py::object(state_tuple(next));
      })
      .def("mask",&Grammar::mask).def("covered",&Grammar::covered).def("clone",&Grammar::clone);
  py::class_<SetGrammar,Grammar,std::shared_ptr<SetGrammar>>(module,"SetGrammar")
      .def(py::init<int64_t,uint64_t,std::vector<int64_t>,std::vector<std::vector<std::pair<int64_t,uint64_t>>>>());
  py::class_<PlainGrammar,Grammar,std::shared_ptr<PlainGrammar>>(module,"PlainGrammar")
      .def(py::init<int64_t,std::vector<int64_t>,std::vector<int64_t>,py::array_t<bool,py::array::c_style|py::array::forcecast>>());
  module.def("route_advance",&route_advance);
  module.def("route_interleave",&interleave,py::call_guard<py::gil_scoped_release>());
  module.def("route_split",&split,py::call_guard<py::gil_scoped_release>());
  module.def("route_interleave_batch",&interleave_batch);
  module.def("route_decode_batch",&decode_batch);
  module.def("route_complete",&complete);
}
