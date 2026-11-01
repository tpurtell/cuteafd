// Resident owner and preallocated MiMo driver. Pointwise/attention arithmetic
// is donated from Hugh Madden's mimo26f-afd v1.3.0 (see vision.cu).
#include "vision.cu"
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>
#include <new>
#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
#include "cuteafd_vision_attention_internal.h"
#endif

static_assert(sizeof(cuteafd_vision_block) == 96, "vision block ABI");
static_assert(offsetof(cuteafd_vision_spec, blocks) == 72, "vision blocks ABI");
static_assert(offsetof(cuteafd_vision_spec, hidden) == 80, "vision geometry ABI");
static_assert(offsetof(cuteafd_vision_spec, patch_bias) == 120, "vision bias ABI");
static_assert(offsetof(cuteafd_vision_spec, norm1_bias) == 224, "vision norm ABI");
static_assert(sizeof(cuteafd_vision_spec) == 256, "vision pointer/count ABI");

namespace {
constexpr size_t BLAS_BYTES = 4 * 1024 * 1024;
constexpr int HIDDEN = 1280, INTER = 4608, MERGED = 5120, PATCH = 1536, CHUNK = 4096;
size_t aligned(size_t n) { return (n + 255) & ~size_t(255); }
struct Owner {
  cuteafd_vision_spec spec{};
  std::vector<cuteafd_vision_block> blocks;
  std::vector<uint64_t> norm1_bias, norm2_bias, q_norm, k_norm;
  cuteafd_vision_ledger ledger{};
  int device = 0;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  unsigned char *weights = nullptr, *arena = nullptr;
  float *x = nullptr, *xt = nullptr, *qkv = nullptr, *proj = nullptr, *gu = nullptr, *fc1 = nullptr, *fc2 = nullptr;
  uint16_t *patches = nullptr, *norm = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr;
  uint16_t *mlp = nullptr, *gelu = nullptr, *output = nullptr;
  uint8_t* rgb = nullptr;
  float* lut = nullptr;
  int32_t *hw_row = nullptr, *hw_col = nullptr, *col = nullptr, *inverse = nullptr;
  int32_t *pos_indices = nullptr, *cu_seqlens = nullptr;
  float *pos_weights = nullptr, *lse = nullptr;
  void *attention = nullptr, *workspace = nullptr;
  bool uploaded = false;
};
bool copy_spec(Owner& o, const cuteafd_vision_spec& s) {
  try {
    o.blocks.assign(s.blocks,s.blocks+s.depth);
    o.norm1_bias.assign(s.norm1_bias,s.norm1_bias+s.depth);
    o.norm2_bias.assign(s.norm2_bias,s.norm2_bias+s.depth);
    o.q_norm.assign(s.q_norm,s.q_norm+s.depth);
    o.k_norm.assign(s.k_norm,s.k_norm+s.depth);
  } catch (const std::bad_alloc&) { return false; }
  o.spec=s;
  o.spec.blocks=o.blocks.data(); o.spec.norm1_bias=o.norm1_bias.data();
  o.spec.norm2_bias=o.norm2_bias.data(); o.spec.q_norm=o.q_norm.data();
  o.spec.k_norm=o.k_norm.data();
  return true;
}
int device_ok(Owner* o) {
  int d; auto e = cudaGetDevice(&d); return e ? int(e) : d == o->device ? 0 : int(cudaErrorInvalidDevice);
}
int drain(Owner* o, int first) {
  auto e = cudaStreamSynchronize(o->stream); return first ? first : int(e);
}
int blas_status(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : -int(s); }
size_t scratch(Owner* o, size_t max_tokens, int output_width) {
  size_t cursor = 0, n = max_tokens * 4;
  auto take = [&](auto& ptr, size_t bytes) {
    using P = typename std::remove_reference<decltype(ptr)>::type;
    ptr = o->arena ? reinterpret_cast<P>(o->arena + cursor) : nullptr;
    cursor += aligned(bytes);
  };
  take(o->x,n*HIDDEN*4); take(o->xt,n*HIDDEN*4); take(o->norm,n*HIDDEN*2);
  take(o->patches,n*PATCH*2); take(o->q,n*VQD*2); take(o->k,n*VKV*VD*2);
  take(o->v,n*VKV*VD*2); take(o->attn,n*VQD*2); take(o->output,max_tokens*output_width*2);
  take(o->qkv,CHUNK*VQKV*4); take(o->proj,CHUNK*HIDDEN*4); take(o->gu,CHUNK*2*INTER*4);
  take(o->mlp,CHUNK*INTER*2); take(o->fc1,CHUNK*MERGED*4); take(o->gelu,CHUNK*MERGED*2);
  take(o->fc2,CHUNK*output_width*4);
  take(o->rgb,n*16*16*3); take(o->lut,3*256*4);
  take(o->hw_row,n*2*4); take(o->hw_col,n*2*4); take(o->col,max_tokens*4); take(o->inverse,max_tokens*4);
  take(o->workspace,BLAS_BYTES);
  return cursor;
}
bool extent(const cuteafd_vision_spec& s, uint64_t offset, size_t bytes) {
  return offset % 256 == 0 && offset <= s.weight_bytes && bytes <= s.weight_bytes - offset;
}
bool valid_spec(const cuteafd_vision_spec& s) {
  if (s.abi_version != 1 || s.depth != 28 || s.reserved || s.max_tokens == 0 || s.max_tokens > 4096 ||
      (s.output_width != 4096 && s.output_width != 6144) || !s.weight_bytes || s.weight_bytes > (2ULL << 30)) return false;
  if (!extent(s,s.patch,HIDDEN*PATCH*2) || !extent(s,s.merger_norm,HIDDEN*4) ||
      !extent(s,s.merger_fc1,MERGED*MERGED*2) || !extent(s,s.merger_fc2,s.output_width*MERGED*2) ||
      !extent(s,s.inv_freq,16*4)) return false;
  for (uint32_t layer=0;layer<s.depth;++layer) {
    const auto& b=s.blocks[layer];
    const uint64_t offsets[] = {b.qkv,b.qkv_bias,b.proj,b.proj_bias,b.gate_up,b.gate_up_bias,b.down,b.down_bias,b.norm1,b.norm2};
    const size_t bytes[] = {VQKV*HIDDEN*2,VQKV*4,HIDDEN*VQD*2,HIDDEN*4,2*INTER*HIDDEN*2,2*INTER*4,HIDDEN*INTER*2,HIDDEN*4,HIDDEN*4,HIDDEN*4};
    for(int i=0;i<10;++i) if(!extent(s,offsets[i],bytes[i])) return false;
    if (b.window < 0 || b.window > 64 || (b.column_order != 0 && b.column_order != 1) ||
        (b.key0_bias != UINT64_MAX && !extent(s,b.key0_bias,VH*4))) return false;
  }
  return s.blocks[27].column_order == 0;
}
__global__ void prepare_grid(int32_t* row, int32_t* column, int32_t* col, int32_t* inverse, int gh, int gw) {
  int u = blockIdx.x * blockDim.x + threadIdx.x, uh = gh / 2, uw = gw / 2;
  if(u >= uh*uw) return;
  int src = (u % uh)*uw + u/uh;
  col[u] = src; inverse[src] = u;
  for(int r=0;r<4;++r) {
    row[2*(u*4+r)] = (u/uw)*2 + r/2;
    row[2*(u*4+r)+1] = (u%uw)*2 + r%2;
    column[2*(u*4+r)] = (src/uw)*2 + r/2;
    column[2*(u*4+r)+1] = (src%uw)*2 + r%2;
  }
}
__global__ void rgb_patches(const uint8_t* rgb, const float* lut, uint16_t* patches, int gh, int gw) {
  size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x, total = size_t(gh)*gw*PATCH;
  if(i >= total) return;
  int patch = i/PATCH, d = i%PATCH, c=d/(2*256), pixel=d%256;
  int unit = patch/4, r=patch%4;
  int y=(unit/(gw/2))*32+(r/2)*16+pixel/16;
  int x=(unit%(gw/2))*32+(r%2)*16+pixel%16;
  patches[i] = bf16_bits(lut[c*256+rgb[(size_t(y)*gw*16+x)*3+c]]);
}
__global__ void cast_bf16(const float* x, uint16_t* y, size_t n) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n)y[i]=bf16_bits(x[i]);
}
__global__ void cast_float(const uint16_t* x, float* y, size_t n) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n)y[i]=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(x+i));
}
int gemm(Owner* o, const uint16_t* x, uint64_t offset, int m, int n, int k, float* y) {
  const float one=1,zero=0;
  // Exclusive handle/workspace and stream, atomics/reduced reduction disabled. The
  // cuBLAS version participates in the encoder identity at integration time.
  return blas_status(cublasGemmEx(o->blas,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,
    o->weights+offset,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&zero,y,CUDA_R_32F,n,
    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
int observe(Owner* o, cuteafd_vision_observer cb, void* ctx, int stage, const void* data, int rows, int width, bool col) {
  if(!cb)return 0;
  int e=drain(o,0); return e ? e : cb(ctx,stage,data,rows,width,int(col));
}
__global__ void vision_lengths(int32_t* lengths,int n) { lengths[0]=0;lengths[1]=n; }
int mha_attention(Owner* o,int n) {
#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
  vision_lengths<<<1,1,0,o->stream>>>(o->cu_seqlens,n);
  int e=cudaGetLastError();if(e)return e;
  return cuteafd_vision_attention_launch(o->attention,o->q,o->k,o->v,o->attn,
    o->lse,o->cu_seqlens,1.f/sqrtf(float(o->spec.head_dim)),o->stream);
#else
  return cudaErrorNotSupported;
#endif
}
#include "vision_qwen.cuh"
#if __has_include("vision_glm.cuh")
#include "vision_glm.cuh"
#define CUTEAFD_HAVE_GLM_VISION 1
#endif
bool valid_tower(const cuteafd_vision_spec& s) {
  if(s.storage_abi!=3 || !s.depth || !s.blocks || !s.norm1_bias ||
     !s.norm2_bias || !s.q_norm || !s.k_norm)return false;
  if(s.abi_version==1)return valid_spec(s);
#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
  if(s.reserved==2)return valid_qwen(s);
#ifdef CUTEAFD_HAVE_GLM_VISION
  if(s.reserved==3)return valid_glm(s);
#endif
#endif
  return false;
}
size_t scratch_tower(Owner* o) {
  const auto& s=o->spec;
  if(s.abi_version==2 && s.reserved==2)return scratch_qwen(o,s.max_tokens,s.output_width);
#ifdef CUTEAFD_HAVE_GLM_VISION
  if(s.abi_version==2 && s.reserved==3)return scratch_glm(o,s.max_tokens,s.output_width);
#endif
  return scratch(o,s.max_tokens,s.output_width);
}
}

extern "C" int32_t cuteafd_vision_required(const cuteafd_vision_spec* s, cuteafd_vision_ledger* out) {
  if(!s || !out || !valid_tower(*s))return cudaErrorInvalidValue;
  Owner o;
  if(!copy_spec(o,*s))return cudaErrorMemoryAllocation;
  size_t bytes=scratch_tower(&o);
  *out={s->weight_bytes,bytes-BLAS_BYTES,BLAS_BYTES,0,0}; return 0;
}
extern "C" int32_t cuteafd_vision_destroy(void* owner) {
  if(!owner)return cudaErrorInvalidValue;
  auto* o=static_cast<Owner*>(owner);
  // Restore the owning device before draining/releasing all storage.
  int previous=-1; cudaGetDevice(&previous); cudaSetDevice(o->device);
  int first=o->stream ? drain(o,0) : 0;
#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
  if(o->attention) { int rc=cuteafd_vision_attention_destroy(o->attention);if(!first)first=rc; }
#endif
  if(o->blas)cublasDestroy(o->blas);
  if(o->weights)cudaFree(o->weights);
  if(o->arena)cudaFree(o->arena);
  if(o->stream)cudaStreamDestroy(o->stream);
  if(previous>=0 && previous!=o->device)cudaSetDevice(previous);
  delete o; return first;
}
extern "C" int32_t cuteafd_vision_create(const cuteafd_vision_spec* s, int32_t device, uint64_t admitted, void** out) {
  if(!out)return cudaErrorInvalidValue; *out=nullptr;
  cuteafd_vision_ledger ledger{}; int e=cuteafd_vision_required(s,&ledger);
  if(e)return e;
  if(admitted < ledger.weights+ledger.scratch+ledger.blas_workspace)return cudaErrorMemoryAllocation;
  e=cudaSetDevice(device);if(e)return e;
  auto* o=new(std::nothrow)Owner; if(!o)return cudaErrorMemoryAllocation;
  // Registration owns all per-layer metadata before the Rust borrow ends.
  if(!copy_spec(*o,*s)){delete o;return cudaErrorMemoryAllocation;}
  o->ledger=ledger; o->device=device;
  auto fail=[&](int rc){cuteafd_vision_destroy(o);return rc;};
  int low,high; e=cudaDeviceGetStreamPriorityRange(&low,&high);if(e)return fail(e);
  e=cudaStreamCreateWithPriority(&o->stream,cudaStreamNonBlocking,low);if(e)return fail(e);
  e=cudaMalloc(reinterpret_cast<void**>(&o->weights),s->weight_bytes);if(e)return fail(e);++o->ledger.device_allocations;
  e=cudaMalloc(reinterpret_cast<void**>(&o->arena),ledger.scratch+ledger.blas_workspace);if(e)return fail(e);++o->ledger.device_allocations;
  scratch_tower(o);
#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
  if(s->abi_version==2) {
    e=cuteafd_vision_attention_create_entry(s->head_dim,0,"",&o->attention);if(e)return fail(e);
  }
#endif
  e=blas_status(cublasCreate(&o->blas));if(e)return fail(e);
  e=blas_status(cublasSetMathMode(o->blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));if(e)return fail(e);
  e=blas_status(cublasSetAtomicsMode(o->blas,CUBLAS_ATOMICS_NOT_ALLOWED));if(e)return fail(e);
  e=blas_status(cublasSetStream(o->blas,o->stream));if(e)return fail(e);
  e=blas_status(cublasSetWorkspace(o->blas,o->workspace,BLAS_BYTES));if(e)return fail(e);
  if(s->abi_version==1) { e=cudaFuncSetAttribute(vit_attn_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(AttnSmem));if(e)return fail(e); }
  *out=o;return 0;
}
extern "C" int32_t cuteafd_vision_upload(void* owner, uint64_t offset, const void* data, uint64_t bytes) {
  if(!owner || !data)return cudaErrorInvalidValue;auto* o=static_cast<Owner*>(owner);
  if(offset>o->spec.weight_bytes || bytes>o->spec.weight_bytes-offset)return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  e=cudaMemcpyAsync(o->weights+offset,data,bytes,cudaMemcpyHostToDevice,o->stream);
  e=drain(o,e);
  // The safe FFI uploads a complete initialized arena, never a partial model.
  if(!e && offset==0 && bytes==o->spec.weight_bytes)o->uploaded=true;
  return e;
}
extern "C" int32_t cuteafd_vision_get_ledger(void* owner, cuteafd_vision_ledger* out) {
  if(!owner || !out)return cudaErrorInvalidValue; *out=static_cast<Owner*>(owner)->ledger;return 0;
}
extern "C" int32_t cuteafd_vision_encode(void* owner, const uint8_t* rgb, uint64_t rgb_bytes, const float* lut,
  int32_t gh, int32_t gw, uint16_t* output, uint64_t output_bytes, cuteafd_vision_observer cb, void* ctx) {
  if(!owner || !rgb || !lut || !output)return cudaErrorInvalidValue;
  auto* o=static_cast<Owner*>(owner); const auto& s=o->spec;
  if(s.abi_version==2 && s.reserved==2)return encode_qwen(o,rgb,rgb_bytes,lut,gh,gw,output,output_bytes,cb,ctx);
#ifdef CUTEAFD_HAVE_GLM_VISION
  if(s.abi_version==2 && s.reserved==3)return encode_glm(o,rgb,rgb_bytes,lut,gh,gw,output,output_bytes,cb,ctx);
#endif
  int64_t count=int64_t(gh)*gw;
  if(!o->uploaded || gh<2 || gw<2 || gh%2 || gw%2 || count>int64_t(s.max_tokens)*4 ||
     rgb_bytes!=uint64_t(count)*16*16*3 || output_bytes!=uint64_t(count/4)*s.output_width*2)return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  int n=int(count),units=n/4;auto stream=o->stream;
  // Every failing launch/observer/BLAS call drains before storage can be reused.
#define RUN(expr) do { int rc=int(expr); if(rc)return drain(o,rc); } while(0)
#define LAUNCH(expr) do { expr; RUN(cudaGetLastError()); } while(0)
  RUN(cudaMemcpyAsync(o->rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  RUN(cudaMemcpyAsync(o->lut,lut,3*256*4,cudaMemcpyHostToDevice,stream));
  LAUNCH((prepare_grid<<<(units+255)/256,256,0,stream>>>(o->hw_row,o->hw_col,o->col,o->inverse,gh,gw)));
  LAUNCH((rgb_patches<<<(size_t(n)*PATCH+255)/256,256,0,stream>>>(o->rgb,o->lut,o->patches,gh,gw)));
  float* x=o->x;float* xt=o->xt;bool col=false;
  for(int r0=0;r0<n;r0+=CHUNK)RUN(gemm(o,o->patches+size_t(r0)*PATCH,s.patch,std::min(CHUNK,n-r0),HIDDEN,PATCH,x+size_t(r0)*HIDDEN));
  RUN(observe(o,cb,ctx,0,x,n,HIDDEN,false));
  auto w=[&](uint64_t off){return reinterpret_cast<const float*>(o->weights+off);};
  for(uint32_t i=0;i<s.depth;++i) {
    const auto& b=s.blocks[i];
    if(bool(b.column_order)!=col) {
      LAUNCH((gather_units_kernel<<<grid_for(long(n)*HIDDEN),256,0,stream>>>(x,xt,b.column_order?o->col:o->inverse,long(n)*HIDDEN,HIDDEN)));
      std::swap(x,xt);col=b.column_order;
    }
    LAUNCH((rmsnorm_bf16_kernel<<<n,256,0,stream>>>(x,w(b.norm1),o->norm,HIDDEN,1e-6f)));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      RUN(gemm(o,o->norm+size_t(r0)*HIDDEN,b.qkv,rows,VQKV,HIDDEN,o->qkv));
      LAUNCH((rope_qkv_kernel<<<rows,256,0,stream>>>(o->qkv,w(b.qkv_bias),col?o->hw_col:o->hw_row,w(s.inv_freq),r0,
        reinterpret_cast<__half*>(o->q),reinterpret_cast<__half*>(o->k),reinterpret_cast<__half*>(o->v))));
    }
    LAUNCH((vit_attn_kernel<<<dim3((n+AT-1)/AT,VKV),ATHREADS,sizeof(AttnSmem),stream>>>(reinterpret_cast<const __half*>(o->q),
      reinterpret_cast<const __half*>(o->k),reinterpret_cast<const __half*>(o->v),n,b.window,b.key0_bias==UINT64_MAX?nullptr:w(b.key0_bias),o->attn)));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      RUN(gemm(o,o->attn+size_t(r0)*VQD,b.proj,rows,HIDDEN,VQD,o->proj));
      LAUNCH((add_bias_residual_kernel<<<grid_for(long(rows)*HIDDEN),256,0,stream>>>(x+size_t(r0)*HIDDEN,o->proj,w(b.proj_bias),long(rows)*HIDDEN,HIDDEN)));
    }
    LAUNCH((rmsnorm_bf16_kernel<<<n,256,0,stream>>>(x,w(b.norm2),o->norm,HIDDEN,1e-6f)));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      RUN(gemm(o,o->norm+size_t(r0)*HIDDEN,b.gate_up,rows,2*INTER,HIDDEN,o->gu));
      LAUNCH((swiglu_bias_bf16_kernel<<<grid_for(long(rows)*INTER),256,0,stream>>>(o->gu,w(b.gate_up_bias),o->mlp,rows,INTER,0.f)));
      RUN(gemm(o,o->mlp,b.down,rows,HIDDEN,INTER,o->proj));
      LAUNCH((add_bias_residual_kernel<<<grid_for(long(rows)*HIDDEN),256,0,stream>>>(x+size_t(r0)*HIDDEN,o->proj,w(b.down_bias),long(rows)*HIDDEN,HIDDEN)));
    }
    if((i+1)%4==0 || i==27)RUN(observe(o,cb,ctx,i+1,x,n,HIDDEN,col));
  }
  LAUNCH((layernorm_bf16_kernel<<<n,256,0,stream>>>(x,w(s.merger_norm),nullptr,o->norm,HIDDEN,1e-6f)));
  if(cb) {
    LAUNCH((cast_float<<<(size_t(n)*HIDDEN+255)/256,256,0,stream>>>(o->norm,xt,size_t(n)*HIDDEN)));
    RUN(observe(o,cb,ctx,29,xt,n,HIDDEN,false));
  }
  for(int u0=0;u0<units;u0+=CHUNK) {
    int rows=std::min(CHUNK,units-u0);
    RUN(gemm(o,o->norm+size_t(u0)*MERGED,s.merger_fc1,rows,MERGED,MERGED,o->fc1));
    LAUNCH((gelu_bf16_kernel<<<grid_for(long(rows)*MERGED),256,0,stream>>>(o->fc1,o->gelu,long(rows)*MERGED,0)));
    RUN(gemm(o,o->gelu,s.merger_fc2,rows,s.output_width,MERGED,o->fc2));
    LAUNCH((cast_bf16<<<(size_t(rows)*s.output_width+255)/256,256,0,stream>>>(o->fc2,o->output+size_t(u0)*s.output_width,size_t(rows)*s.output_width)));
  }
  RUN(observe(o,cb,ctx,30,o->output,units,s.output_width,false));
  RUN(cudaMemcpyAsync(output,o->output,output_bytes,cudaMemcpyDeviceToHost,stream));
  RUN(cudaStreamSynchronize(stream));++o->ledger.encodes;
#undef LAUNCH
#undef RUN
  return 0;
}

extern "C" int32_t cuteafd_vision_rmsnorm_bf16(const float* x,const float* w,uint16_t* y,int32_t rows,int32_t dim,float eps,void* stream) {
  if(!x||!w||!y||rows<=0||dim<=0||dim>16384||eps<=0)return cudaErrorInvalidValue;
  rmsnorm_bf16_kernel<<<rows,256,0,static_cast<cudaStream_t>(stream)>>>(x,w,y,dim,eps);return cudaGetLastError();
}
extern "C" int32_t cuteafd_vision_layernorm_bf16(const float* x,const float* w,const float* bias,uint16_t* y,int32_t rows,int32_t dim,float eps,void* stream) {
  if(!x||!w||!y||rows<=0||dim<=0||dim>16384||eps<=0)return cudaErrorInvalidValue;
  layernorm_bf16_kernel<<<rows,256,0,static_cast<cudaStream_t>(stream)>>>(x,w,bias,y,dim,eps);return cudaGetLastError();
}
extern "C" int32_t cuteafd_vision_swiglu_bf16(const float* x,const float* b,uint16_t* y,int64_t rows,int32_t dim,float clamp,void* stream) {
  if(!x||!y||rows<=0||rows>16384||dim<=0||dim>16384||clamp<0)return cudaErrorInvalidValue;
  swiglu_bias_bf16_kernel<<<grid_for(rows*dim),256,0,static_cast<cudaStream_t>(stream)>>>(x,b,y,rows,dim,clamp);return cudaGetLastError();
}
extern "C" int32_t cuteafd_vision_gelu_bf16(const float* x,uint16_t* y,int64_t n,int32_t tanh_mode,void* stream) {
  if(!x||!y||n<=0||n>int64_t(16384)*16384||(tanh_mode!=0&&tanh_mode!=1))return cudaErrorInvalidValue;
  gelu_bf16_kernel<<<grid_for(n),256,0,static_cast<cudaStream_t>(stream)>>>(x,y,n,tanh_mode);return cudaGetLastError();
}
