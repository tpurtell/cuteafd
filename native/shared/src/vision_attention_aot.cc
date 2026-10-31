#include "cuteafd_vision_attention_internal.h"
#include "vision_attention_d64.h"
#include "vision_attention_d72.h"
#include "vision_attention_d128_full.h"
#include "vision_axial_rotary_d128_bf16.h"
#include "vision_rms_bf16_gamma_h2048.h"
#include <mutex>
#include <new>

namespace {
cuteafd_vision_attention_d64_Kernel_Module_t module64{};
cuteafd_vision_attention_d72_Kernel_Module_t module72{};
cuteafd_vision_attention_d128_full_Kernel_Module_t module128{};
cuteafd_vision_axial_rotary_d128_bf16_Kernel_Module_t rotary128{};
cuteafd_vision_rms_bf16_gamma_h2048_Kernel_Module_t rms2048{};
std::mutex mutex;
uint32_t references[3]{};
bool loaded[3][32]{};
bool pointwise_loaded[32]{};
// Head 128 (Pixtral) loads its attention and its two pointwise programs together.
int load_d128(int device) {
  int result=0;
  if(!module128.module) { auto* library=&module128.module; void* args[]={&library,&result};
    _mlir_cuteafd_vision_attention_d128_full_cuda_init(args); if(result)return result; }
  if(!rotary128.module) { auto* library=&rotary128.module; void* args[]={&library,&result};
    _mlir_cuteafd_vision_axial_rotary_d128_bf16_cuda_init(args); if(result)return result; }
  if(!rms2048.module) { auto* library=&rms2048.module; void* args[]={&library,&result};
    _mlir_cuteafd_vision_rms_bf16_gamma_h2048_cuda_init(args); if(result)return result; }
  if(!loaded[2][device]) {
    auto* a=&module128.module; void* args[]={&a,&device,&result};
    _mlir_cuteafd_vision_attention_d128_full_cuda_load_to_device(args); if(result)return result;
    loaded[2][device]=true;
  }
  if(!pointwise_loaded[device]) {
    auto* r=&rotary128.module; void* rotary[]={&r,&device,&result};
    _mlir_cuteafd_vision_axial_rotary_d128_bf16_cuda_load_to_device(rotary); if(result)return result;
    auto* n=&rms2048.module; void* norm[]={&n,&device,&result};
    _mlir_cuteafd_vision_rms_bf16_gamma_h2048_cuda_load_to_device(norm); if(result)return result;
    pointwise_loaded[device]=true;
  }
  return 0;
}
struct Attention { int dim, device; };
}
extern "C" int32_t cuteafd_vision_attention_create(int32_t dim, void** out) {
  if(!out || (dim!=64 && dim!=72 && dim!=128))return cudaErrorInvalidValue;
  *out=nullptr;
  int device=-1;auto status=cudaGetDevice(&device);
  if(status)return status;
  if(device<0 || device>=32)return cudaErrorInvalidDevice;
  auto* owner=new(std::nothrow)Attention{dim,device};
  if(!owner)return cudaErrorMemoryAllocation;
  std::lock_guard<std::mutex> guard(mutex);
  if(dim==128) {
    int result=load_d128(device);
    if(result) { delete owner; return result; }
    ++references[2];*out=owner;return 0;
  }
  int index=dim==72, result=0;
  auto* library=dim==72?&module72.module:&module64.module;
  if(!*library) {
    void* args[]={&library,&result};
    if(dim==72)_mlir_cuteafd_vision_attention_d72_cuda_init(args);
    else _mlir_cuteafd_vision_attention_d64_cuda_init(args);
  }
  if(!result && !loaded[index][device]) {
    void* args[]={&library,&device,&result};
    if(dim==72)_mlir_cuteafd_vision_attention_d72_cuda_load_to_device(args);
    else _mlir_cuteafd_vision_attention_d64_cuda_load_to_device(args);
    if(!result)loaded[index][device]=true;
  }
  if(result) { delete owner; return result; }
  ++references[index];*out=owner;return 0;
}
extern "C" int32_t cuteafd_vision_attention_destroy(void* opaque) {
  if(!opaque)return 0;
  auto* owner=static_cast<Attention*>(opaque);
  // The caller has drained its stream before releasing its module reference.
  std::lock_guard<std::mutex> guard(mutex);
  if(owner->dim==128) {
    int result=0;
    if(--references[2]==0) {
      for(auto* library:{&module128.module,&rotary128.module,&rms2048.module}) {
        int rc=cudaLibraryUnload(*library); if(!result)result=rc; if(!rc)*library=nullptr;
      }
      if(!result) { for(auto& value:loaded[2])value=false; for(auto& value:pointwise_loaded)value=false; }
    }
    delete owner;return result;
  }
  int index=owner->dim==72,result=0;
  if(--references[index]==0) {
    auto* library=index?&module72.module:&module64.module;
    result=cudaLibraryUnload(*library);
    if(!result) { *library=nullptr;for(auto& value:loaded[index])value=false; }
  }
  delete owner;return result;
}
extern "C" int32_t cuteafd_vision_attention_launch(void* opaque,void* q,void* k,void* v,
  void* out,void* lse,void* cu_seqlens,float scale,void* stream) {
  if(!opaque)return cudaErrorInitializationError;
  auto* owner=static_cast<Attention*>(opaque);int device=-1;
  auto status=cudaGetDevice(&device);if(status)return status;
  if(device!=owner->device)return cudaErrorInvalidDevice;
  // Head 128 (Pixtral): one segment per image, cu = [0, n].
  if(owner->dim==128)return cute_dsl_cuteafd_vision_attention_d128_full_wrapper(&module128,
    q,k,v,out,lse,cu_seqlens,scale,static_cast<cudaStream_t>(stream));
  if(owner->dim==72)return cute_dsl_cuteafd_vision_attention_d72_wrapper(&module72,
    q,k,v,out,lse,cu_seqlens,cu_seqlens,nullptr,scale,static_cast<cudaStream_t>(stream));
  return cute_dsl_cuteafd_vision_attention_d64_wrapper(&module64,
    q,k,v,out,lse,cu_seqlens,cu_seqlens,nullptr,scale,static_cast<cudaStream_t>(stream));
}

// Pixtral pointwise programs; valid only while a head-128 owner holds them loaded.
extern "C" int32_t cuteafd_vision_axial_rotary_d128(void* opaque,void* q,void* k,void* cos,void* sin,
  void* out_q,void* out_k,int32_t rows,void* stream) {
  if(!opaque || static_cast<Attention*>(opaque)->dim!=128)return cudaErrorInitializationError;
  return cute_dsl_cuteafd_vision_axial_rotary_d128_bf16_wrapper(&rotary128,q,k,cos,sin,out_q,out_k,
    rows,static_cast<cudaStream_t>(stream));
}
extern "C" int32_t cuteafd_vision_rms_h2048(void* opaque,void* x,void* weight,void* out,int32_t rows,void* stream) {
  if(!opaque || static_cast<Attention*>(opaque)->dim!=128)return cudaErrorInitializationError;
  // bias/cos/sin/positions are ignored by the BF16-before-gamma program.
  return cute_dsl_cuteafd_vision_rms_bf16_gamma_h2048_wrapper(&rms2048,x,weight,weight,weight,weight,weight,out,
    rows,static_cast<cudaStream_t>(stream));
}
