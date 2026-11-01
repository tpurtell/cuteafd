// Shared BF16 causal PCM pyramid. Arithmetic comes solely from pinned AOT
// programs; this owner handles admission, streaming history and lifetimes.
#include "cuteafd_pcm_audio.h"
#include "cuteafd_pcm_support_internal.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <new>
#include <type_traits>

namespace {
constexpr uint32_t WIDTHS[4]={256,512,1024,2048}, PATCHES[4]={16,6,5,4};
constexpr uint32_t STRIDE=1920, CHUNK_TOKENS=16, MAX_SAMPLES=7200000;
uint64_t aligned(uint64_t n){return (n+255)&~uint64_t(255);}
struct Owner {
  cuteafd_pcm_spec s{};
  cuteafd_pcm_ledger ledger{};
  int device=0;
  cudaStream_t stream=nullptr;
  void* modules=nullptr;
  unsigned char *weights=nullptr,*arena=nullptr;
  uint16_t *x=nullptr,*y=nullptr,*z=nullptr,*fc=nullptr,*gelu=nullptr,*history[4][2]{};
  uint64_t uploaded=0;
};
int drain(Owner* o,int first){int e=cudaStreamSynchronize(o->stream);return first?first:e;}
bool extent(const cuteafd_pcm_spec& s,uint64_t off,uint64_t bytes){
  return off%256==0&&off<=s.weight_bytes&&bytes<=s.weight_bytes-off;
}
bool valid(const cuteafd_pcm_spec& s){
  if(s.abi_version!=1||s.reserved||!s.max_samples||s.max_samples>MAX_SAMPLES||
     s.output_width!=8192||!s.weight_bytes||s.weight_bytes>(1ULL<<30))return false;
  uint64_t previous=1;
  for(int i=0;i<4;++i){
    uint64_t h=WIDTHS[i];const auto& a=s.stages[i];
    if(!extent(s,a.projection,h*previous*PATCHES[i]*2)||!extent(s,a.bias,h*2)||!extent(s,a.norm,h*2))return false;
    for(const auto& b:a.blocks){
      const uint64_t offsets[]={b.conv,b.conv_bias,b.norm,b.fc1,b.fc1_bias,b.fc2,b.fc2_bias};
      const uint64_t sizes[]={h*7*2,h*2,h*2,h*h*4*2,h*4*2,h*h*4*2,h*2};
      for(int j=0;j<7;++j)if(!extent(s,offsets[j],sizes[j]))return false;
    }
    previous=h;
  }
  return extent(s,s.norm,2048*2)&&extent(s,s.projection,2048ULL*s.output_width*2)&&
    extent(s,s.bias,s.output_width*2)&&extent(s,s.post_norm,s.output_width*2);
}
uint64_t scratch(Owner* o){
  uint64_t tokens=std::min(CHUNK_TOKENS,(o->s.max_samples+STRIDE-1)/STRIDE);
  uint64_t rows=tokens*STRIDE,largest=tokens*o->s.output_width;
  for(int i=0;i<4;++i){rows/=PATCHES[i];largest=std::max(largest,rows*WIDTHS[i]);}
  uint64_t cursor=0;
  auto take=[&](uint16_t*& p,uint64_t n){p=o->arena?reinterpret_cast<uint16_t*>(o->arena+cursor):nullptr;cursor+=aligned(n*2);};
  take(o->x,largest);take(o->y,largest);take(o->z,largest);
  take(o->fc,largest*4);take(o->gelu,largest*4);
  for(int i=0;i<4;++i)for(int j=0;j<2;++j)take(o->history[i][j],WIDTHS[i]*6);
  return cursor;
}
void* weight(Owner* o,uint64_t offset){return o->weights+offset;}
int call(Owner* o,int op,int width,int outputs,void** args,int rows){
  return cuteafd_pcm_support_launch(o->modules,op,width,outputs,args,rows,o->stream);
}
int linear(Owner* o,void* x,void* y,uint64_t w,uint64_t b,int rows,int k,int n){
  void* args[]={x,weight(o,w),weight(o,b),y};return call(o,0,k,n,args,rows);
}
int norm(Owner* o,void* x,void* y,uint64_t w,int rows,int width){
  void* gamma=weight(o,w);void* args[]={x,gamma,gamma,gamma,gamma,gamma,y};
  return call(o,2,width,0,args,rows);
}
int observe(Owner* o,cuteafd_pcm_observer cb,void* context,int stage,uint32_t offset,void* x,int rows,int width){
  if(!cb)return 0;int e=drain(o,0);return e?e:cb(context,stage,offset,x,rows,width);
}
}
extern "C" int32_t cuteafd_pcm_required(const cuteafd_pcm_spec* spec,cuteafd_pcm_ledger* out){
  if(!spec||!out||!valid(*spec))return cudaErrorInvalidValue;
  Owner o;o.s=*spec;*out={spec->weight_bytes,scratch(&o),2,0};return 0;
}
extern "C" int32_t cuteafd_pcm_destroy(void* p){
  if(!p)return 0;auto* o=static_cast<Owner*>(p);int device=-1;int first=cudaGetDevice(&device);
  if(!first&&device!=o->device)return cudaErrorInvalidDevice;
  auto save=[&](int e){if(!first)first=e;};
  if(o->stream)save(cudaStreamSynchronize(o->stream));
  if(o->modules)save(cuteafd_pcm_support_destroy(o->modules));
  if(o->arena)save(cudaFree(o->arena));if(o->weights)save(cudaFree(o->weights));
  if(o->stream)save(cudaStreamDestroy(o->stream));delete o;return first;
}
extern "C" int32_t cuteafd_pcm_create(const cuteafd_pcm_spec* spec,int32_t device,uint64_t admitted,void** out){
  if(!out)return cudaErrorInvalidValue;*out=nullptr;cuteafd_pcm_ledger ledger{};
  int e=cuteafd_pcm_required(spec,&ledger);if(e)return e;
  if(admitted<ledger.weights+ledger.scratch)return cudaErrorMemoryAllocation;
  int current=-1;e=cudaGetDevice(&current);if(e)return e;if(current!=device)return cudaErrorInvalidDevice;
  auto* o=new(std::nothrow)Owner{};if(!o)return cudaErrorMemoryAllocation;
  o->s=*spec;o->ledger=ledger;o->device=device;
  int least=0,greatest=0;e=cudaDeviceGetStreamPriorityRange(&least,&greatest);
  if(!e)e=cudaStreamCreateWithPriority(&o->stream,cudaStreamNonBlocking,least);
  if(!e)e=cudaMalloc(&o->weights,ledger.weights);
  if(!e)e=cudaMalloc(&o->arena,ledger.scratch);
  if(!e){scratch(o);e=cuteafd_pcm_support_create(&o->modules);}
  if(e){cuteafd_pcm_destroy(o);return e;}*out=o;return 0;
}
extern "C" int32_t cuteafd_pcm_upload(void* p,uint64_t offset,const void* data,uint64_t bytes){
  if(!p||!data||!bytes)return cudaErrorInvalidValue;auto* o=static_cast<Owner*>(p);
  // Sequential uploads cover the complete aligned arena, including padding.
  if(offset!=o->uploaded||offset>o->s.weight_bytes||bytes>o->s.weight_bytes-offset)return cudaErrorInvalidValue;
  int d=-1;int e=cudaGetDevice(&d);if(e)return e;if(d!=o->device)return cudaErrorInvalidDevice;
  e=cudaMemcpyAsync(o->weights+offset,data,bytes,cudaMemcpyHostToDevice,o->stream);e=drain(o,e);
  if(!e)o->uploaded+=bytes;return e;
}
extern "C" int32_t cuteafd_pcm_get_ledger(void* p,cuteafd_pcm_ledger* out){
  if(!p||!out)return cudaErrorInvalidValue;*out=static_cast<Owner*>(p)->ledger;return 0;
}
extern "C" int32_t cuteafd_pcm_encode(void* p,const uint16_t* pcm,uint32_t samples,uint16_t* output,
    uint64_t output_bytes,cuteafd_pcm_observer cb,void* context){
  if(!p||!pcm||!output)return cudaErrorInvalidValue;auto* o=static_cast<Owner*>(p);
  if(!samples||samples>o->s.max_samples||o->uploaded!=o->s.weight_bytes)return cudaErrorInvalidValue;
  uint32_t tokens=(samples+STRIDE-1)/STRIDE;
  if(output_bytes<uint64_t(tokens)*o->s.output_width*2)return cudaErrorInvalidValue;
  int d=-1;int e=cudaGetDevice(&d);if(e)return e;if(d!=o->device)return cudaErrorInvalidDevice;
  for(int i=0;i<4&&!e;++i)for(int j=0;j<2&&!e;++j)e=cudaMemsetAsync(o->history[i][j],0,WIDTHS[i]*6*2,o->stream);
  for(uint32_t offset=0;offset<tokens&&!e;offset+=CHUNK_TOKENS){
    uint32_t count=std::min(CHUNK_TOKENS,tokens-offset),rows=count*STRIDE,previous=1;
    e=cudaMemcpyAsync(o->x,pcm+uint64_t(offset)*STRIDE,uint64_t(rows)*2,cudaMemcpyHostToDevice,o->stream);
    for(int i=0;i<4&&!e;++i){
      const auto& a=o->s.stages[i];uint32_t h=WIDTHS[i];rows/=PATCHES[i];
      e=linear(o,o->x,o->y,a.projection,a.bias,rows,previous*PATCHES[i],h);
      if(!e)e=norm(o,o->y,o->x,a.norm,rows,h);
      if(!e)e=observe(o,cb,context,i*3,offset,o->x,rows,h);
      for(int j=0;j<2&&!e;++j){
        const auto& b=a.blocks[j];void* conv[]={o->x,weight(o,b.conv),weight(o,b.conv_bias),o->history[i][j],o->y,o->history[i][j]};
        e=call(o,1,h,0,conv,rows);
        if(!e)e=norm(o,o->y,o->z,b.norm,rows,h);
        if(!e)e=linear(o,o->z,o->fc,b.fc1,b.fc1_bias,rows,h,h*4);
        void* gelu[]={o->fc,o->gelu};if(!e)e=call(o,3,h*4,0,gelu,rows);
        if(!e)e=linear(o,o->gelu,o->y,b.fc2,b.fc2_bias,rows,h*4,h);
        void* residual[]={o->x,o->y,o->x};if(!e)e=call(o,4,h,0,residual,rows);
        if(!e)e=observe(o,cb,context,i*3+j+1,offset,o->x,rows,h);
      }
      previous=h;
    }
    if(!e)e=norm(o,o->x,o->z,o->s.norm,rows,previous);
    if(!e)e=linear(o,o->z,o->y,o->s.projection,o->s.bias,rows,previous,o->s.output_width);
    if(!e)e=norm(o,o->y,o->z,o->s.post_norm,rows,o->s.output_width);
    if(!e)e=observe(o,cb,context,12,offset,o->z,rows,o->s.output_width);
    if(!e)e=cudaMemcpyAsync(output+uint64_t(offset)*o->s.output_width,o->z,uint64_t(count)*o->s.output_width*2,cudaMemcpyDeviceToHost,o->stream);
    // Drain even on observer/launch errors before reusing host or device storage.
    e=drain(o,e);
  }
  e=drain(o,e);if(!e)++o->ledger.encodes;return e;
}
