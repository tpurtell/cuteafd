// Mistral Large 4's bundled Pixtral tower, official BF16 arithmetic
// (transformers pixtral + mistral_large4): every Linear is BF16 x BF16 with F32
// accumulation and one BF16 store; the residual stream is BF16; RMS norms,
// axial rotary and attention are the #936 AOT programs (head 128, h2048).
// Included after the shared Owner/toolkit.
//
// Layout: patches are row-major over the image grid (HF flattens [d,h,w]);
// merged rows are row-major over (h/2, w/2), the 2x2 unfold reading channel
// major (c * 4 + kh * 2 + kw). The LM span is per merged row [IMG] x w then
// [IMG_BREAK], the last [IMG_END]: merged rows land in the [IMG] slots and the
// checkpoint's own marker embedding rows (merger_extra[4], [5]) in the others.
constexpr int MI_HIDDEN=2048, MI_HEADS=16, MI_HEAD=128, MI_INTER=6144;
constexpr int MI_PATCH=588, MI_MERGED=8192, MI_LM=6144, MI_MAX_SIDE=55;

size_t scratch_mistral(Owner* o, size_t max_merged, int output_width) {
  size_t cursor=0,n=max_merged*4,span=max_merged+MI_MAX_SIDE;
  auto take=[&](auto& ptr,size_t bytes) {
    using P=typename std::remove_reference<decltype(ptr)>::type;
    ptr=o->arena ? reinterpret_cast<P>(o->arena+cursor) : nullptr;
    cursor+=aligned(bytes);
  };
  // x / xt: F32 GEMM outputs; xt is reused as the BF16 merger input after the tower.
  take(o->x,n*MI_HIDDEN*4);take(o->xt,n*MI_HIDDEN*4);take(o->norm,n*MI_HIDDEN*2);
  take(o->patches,n*MI_PATCH*2);
  take(o->q,n*MI_HIDDEN*2);take(o->k,n*MI_HIDDEN*2);take(o->v,n*MI_HIDDEN*2);
  take(o->attn,n*MI_HIDDEN*2);take(o->output,span*size_t(output_width)*2);
  take(o->qkv,size_t(CHUNK)*3*MI_HIDDEN*4);take(o->proj,size_t(CHUNK)*MI_HIDDEN*4);
  take(o->gu,size_t(CHUNK)*2*MI_INTER*4);take(o->mlp,size_t(CHUNK)*MI_INTER*2);
  take(o->fc1,size_t(CHUNK)*MI_LM*4);take(o->gelu,size_t(CHUNK)*MI_LM*2);
  take(o->fc2,size_t(CHUNK)*output_width*4);
  take(o->rgb,n*MI_PATCH);take(o->lut,3*256*4);
  // Axial positions (h, w) per patch; the rotary tables reuse proj/qkv per chunk.
  take(o->hw_row,n*2*4);
  // AOT attention's LSE strides are fixed at its 16K patch capacity.
  take(o->lse,size_t(16384)*MI_HEADS*4);take(o->cu_seqlens,2*4);
  take(o->workspace,BLAS_BYTES);return cursor;
}

bool valid_mistral(const cuteafd_vision_spec& s) {
  if(s.abi_version!=2 || s.reserved!=4 || s.geometry_reserved || !s.max_tokens ||
     s.max_tokens>uint32_t(MI_MAX_SIDE*MI_MAX_SIDE) || s.hidden!=2048 || s.depth!=30 || s.heads!=16 ||
     s.kv_heads!=16 || s.head_dim!=128 || s.intermediate!=6144 || s.patch_size!=14 ||
     s.merger_width!=8192 || s.output_width!=6144 || s.norm_eps!=1e-5f || !s.weight_bytes ||
     s.weight_bytes>(4ULL<<30))return false;
  if(!extent(s,s.patch,size_t(2048)*588*2) || !extent(s,s.inv_freq,64*4) ||
     !extent(s,s.merger_norm,2048*2) || !extent(s,s.merger_fc1,size_t(6144)*2048*2) ||
     !extent(s,s.merger_fc2,size_t(6144)*6144*2))return false;
  const size_t mb[]={2048*2,size_t(2048)*8192*2,0,0,6144*2,6144*2,0,0};
  for(int i=0;i<8;++i)if(mb[i] && !extent(s,s.merger_extra[i],mb[i]))return false;
  for(int i=0;i<30;++i) {
    const auto& b=s.blocks[i];
    const uint64_t offsets[]={b.qkv,b.proj,b.gate_up,b.down,b.norm1,b.norm2};
    const size_t bytes[]={size_t(3)*2048*2048*2,size_t(2048)*2048*2,size_t(2)*6144*2048*2,
      size_t(2048)*6144*2,2048*2,2048*2};
    for(int j=0;j<6;++j)if(!extent(s,offsets[j],bytes[j]))return false;
    if(b.qkv_bias!=UINT64_MAX || b.proj_bias!=UINT64_MAX || b.gate_up_bias!=UINT64_MAX ||
       b.down_bias!=UINT64_MAX || b.window || b.column_order || b.key0_bias!=UINT64_MAX)return false;
  }
  return true;
}

__device__ __forceinline__ float mi_bf16(uint16_t x) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
}
// RGB8 [gh*14, gw*14, 3] -> BF16 patches [n, c*196 + kh*14 + kw], row-major patches.
__global__ void mi_rgb_patches(const uint8_t* rgb,const float* lut,uint16_t* out,int gh,int gw) {
  size_t total=size_t(gh)*gw*MI_PATCH;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x) {
    int p=i/MI_PATCH,d=i%MI_PATCH,c=d/196,ky=(d%196)/14,kx=d%14;
    int y=(p/gw)*14+ky,x=(p%gw)*14+kx;
    out[i]=bf16_bits(lut[c*256+rgb[(size_t(y)*gw*14+x)*3+c]]);
  }
}
// One BF16 store of an F32 GEMM output (HF Linear in BF16).
__global__ void mi_cast(const float* x,uint16_t* y,size_t n) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)y[i]=bf16_bits(x[i]);
}
// [q | k | v] F32 GEMM rows -> BF16 q, k, v at global rows.
__global__ void mi_split_qkv(const float* qkv,uint16_t* q,uint16_t* k,uint16_t* v,int r0,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
    size_t row=i/MI_HIDDEN,c=i%MI_HIDDEN,at=(size_t(r0)+row)*MI_HIDDEN+c;
    const float* src=qkv+row*3*MI_HIDDEN;
    q[at]=bf16_bits(src[c]);k[at]=bf16_bits(src[MI_HIDDEN+c]);v[at]=bf16_bits(src[2*MI_HIDDEN+c]);
  }
}
// Row-major positions and Pixtral's full BF16 cos/sin rows: angle j < 32 uses h,
// 32..63 uses w (inv_freq interleaved), repeated for 64..127.
__global__ void mi_positions(int32_t* hw,int gh,int gw) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<gh*gw) { hw[2*p]=p/gw; hw[2*p+1]=p%gw; }
}
__global__ void mi_rotary_tables(const int32_t* hw,const float* inv,uint16_t* cos_t,uint16_t* sin_t,int r0,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
    size_t row=i/MI_HEAD;int j=i%MI_HEAD,f=j%64;
    float angle=__fmul_rn(float(hw[2*(r0+row)+(f>=32)]),inv[f]);
    cos_t[i]=bf16_bits(cosf(angle));sin_t[i]=bf16_bits(sinf(angle));
  }
}
// h = BF16(h + y) for a BF16 linear output y (cast first, as HF adds BF16 tensors).
__global__ void mi_residual(uint16_t* h,const float* y,size_t r0,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
    size_t at=r0*MI_HIDDEN+i;
    h[at]=bf16_bits(__fadd_rn(mi_bf16(h[at]),mi_bf16(bf16_bits(y[i]))));
  }
}
// BF16(silu(BF16 gate) rounded to BF16, times BF16 up).
__global__ void mi_swiglu(const float* gu,uint16_t* out,size_t rows) {
  size_t count=rows*MI_INTER;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
    size_t row=i/MI_INTER,c=i%MI_INTER;
    float g=mi_bf16(bf16_bits(gu[row*2*MI_INTER+c])),u=mi_bf16(bf16_bits(gu[row*2*MI_INTER+MI_INTER+c]));
    float silu=mi_bf16(bf16_bits(__fdiv_rn(g,__fadd_rn(1.f,expf(-g)))));
    out[i]=bf16_bits(__fmul_rn(silu,u));
  }
}
// HF unfold(kernel 2, stride 2) over [d, h, w]: merged row (uy, ux), feature d*4 + kh*2 + kw.
__global__ void mi_unfold(const uint16_t* norm,uint16_t* out,int gh,int gw) {
  int uw=gw/2;size_t total=size_t(gh/2)*uw*MI_MERGED;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x) {
    size_t u=i/MI_MERGED;int f=i%MI_MERGED,d=f/4,kh=(f/2)%2,kw=f%2;
    size_t p=(size_t(u/uw)*2+kh)*gw+size_t(u%uw)*2+kw;
    out[i]=norm[p*MI_HIDDEN+d];
  }
}
// Exact-erf GELU of a BF16 linear output, one BF16 store.
__global__ void mi_gelu(const float* x,uint16_t* y,size_t n) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
    float a=mi_bf16(bf16_bits(x[i]));
    y[i]=bf16_bits(0.5f*a*(1.f+erff(a*0.7071067811865475f)));
  }
}
// Merged rows (row-major) into the span's [IMG] slots; marker rows elsewhere.
__global__ void mi_span(const uint16_t* merged,const uint16_t* row_break,const uint16_t* end,
  uint16_t* out,int uh,int uw) {
  size_t total=size_t(uh)*(uw+1)*MI_LM;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x) {
    size_t slot=i/MI_LM;int c=i%MI_LM,row=slot/(uw+1),col=slot%(uw+1);
    out[i]=col<uw ? merged[(size_t(row)*uw+col)*MI_LM+c] : (row==uh-1 ? end[c] : row_break[c]);
  }
}

int encode_mistral(Owner* o,const uint8_t* rgb,uint64_t rgb_bytes,const float* lut,int gh,int gw,
  uint16_t* output,uint64_t output_bytes,cuteafd_vision_observer cb,void* ctx) {
  const auto& s=o->spec;int64_t count=int64_t(gh)*gw;
  int uh=gh/2,uw=gw/2;
  if(!o->uploaded || !rgb || !lut || !output || gh<2 || gw<2 || gh%2 || gw%2 ||
     uh>MI_MAX_SIDE || uw>MI_MAX_SIDE || count>int64_t(s.max_tokens)*4 ||
     rgb_bytes!=uint64_t(count)*MI_PATCH ||
     output_bytes!=uint64_t(uh)*(uw+1)*MI_LM*2)return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  int n=int(count),units=uh*uw;auto stream=o->stream;
#define MI_RUN(expr) do { int rc=int(expr);if(rc)return drain(o,rc); } while(0)
#define MI_LAUNCH(expr) do { expr;MI_RUN(cudaGetLastError()); } while(0)
  auto bf=[&](uint64_t offset){return reinterpret_cast<uint16_t*>(o->weights+offset);};
  uint16_t* h=reinterpret_cast<uint16_t*>(o->xt);  // BF16 residual stream (n x 2048)
  MI_RUN(cudaMemcpyAsync(o->rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  MI_RUN(cudaMemcpyAsync(o->lut,lut,768*4,cudaMemcpyHostToDevice,stream));
  MI_LAUNCH((mi_positions<<<(n+255)/256,256,0,stream>>>(o->hw_row,gh,gw)));
  MI_LAUNCH((mi_rgb_patches<<<grid_for(long(n)*MI_PATCH),256,0,stream>>>(o->rgb,o->lut,o->patches,gh,gw)));
  for(int r0=0;r0<n;r0+=CHUNK) {
    int rows=std::min(CHUNK,n-r0);
    MI_RUN(gemm(o,o->patches+size_t(r0)*MI_PATCH,s.patch,rows,MI_HIDDEN,MI_PATCH,o->x));
    MI_LAUNCH((mi_cast<<<grid_for(long(rows)*MI_HIDDEN),256,0,stream>>>(o->x,o->norm+size_t(r0)*MI_HIDDEN,size_t(rows)*MI_HIDDEN)));
  }
  MI_RUN(cuteafd_vision_rms_h2048(o->attention,o->norm,bf(s.merger_extra[0]),h,n,stream));
  if(cb) {
    MI_LAUNCH((cast_float<<<(size_t(n)*MI_HIDDEN+255)/256,256,0,stream>>>(h,o->x,size_t(n)*MI_HIDDEN)));
    MI_RUN(observe(o,cb,ctx,0,o->x,n,MI_HIDDEN,false));
  }
  const float* inv=reinterpret_cast<const float*>(o->weights+s.inv_freq);
  for(int i=0;i<30;++i) {
    const auto& b=s.blocks[i];
    MI_RUN(cuteafd_vision_rms_h2048(o->attention,h,bf(b.norm1),o->norm,n,stream));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      MI_RUN(gemm(o,o->norm+size_t(r0)*MI_HIDDEN,b.qkv,rows,3*MI_HIDDEN,MI_HIDDEN,o->qkv));
      MI_LAUNCH((mi_split_qkv<<<grid_for(long(rows)*MI_HIDDEN),256,0,stream>>>(o->qkv,o->attn,o->k,o->v,r0,size_t(rows)*MI_HIDDEN)));
      // q is staged in attn and rotated into q; k rotates through gu (outputs must not alias).
      uint16_t* cos_t=reinterpret_cast<uint16_t*>(o->proj);
      uint16_t* sin_t=cos_t+size_t(rows)*MI_HEAD;
      MI_LAUNCH((mi_rotary_tables<<<grid_for(long(rows)*MI_HEAD),256,0,stream>>>(o->hw_row,inv,cos_t,sin_t,r0,size_t(rows)*MI_HEAD)));
      uint16_t* k_rot=reinterpret_cast<uint16_t*>(o->gu);
      MI_RUN(cuteafd_vision_axial_rotary_d128(o->attention,o->attn+size_t(r0)*MI_HIDDEN,o->k+size_t(r0)*MI_HIDDEN,
        cos_t,sin_t,o->q+size_t(r0)*MI_HIDDEN,k_rot,rows,stream));
      MI_RUN(cudaMemcpyAsync(o->k+size_t(r0)*MI_HIDDEN,k_rot,size_t(rows)*MI_HIDDEN*2,cudaMemcpyDeviceToDevice,stream));
    }
    MI_RUN(mha_attention(o,n));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      MI_RUN(gemm(o,o->attn+size_t(r0)*MI_HIDDEN,b.proj,rows,MI_HIDDEN,MI_HIDDEN,o->proj));
      MI_LAUNCH((mi_residual<<<grid_for(long(rows)*MI_HIDDEN),256,0,stream>>>(h,o->proj,r0,size_t(rows)*MI_HIDDEN)));
    }
    MI_RUN(cuteafd_vision_rms_h2048(o->attention,h,bf(b.norm2),o->norm,n,stream));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      MI_RUN(gemm(o,o->norm+size_t(r0)*MI_HIDDEN,b.gate_up,rows,2*MI_INTER,MI_HIDDEN,o->gu));
      MI_LAUNCH((mi_swiglu<<<grid_for(long(rows)*MI_INTER),256,0,stream>>>(o->gu,o->mlp,rows)));
      MI_RUN(gemm(o,o->mlp,b.down,rows,MI_HIDDEN,MI_INTER,o->proj));
      MI_LAUNCH((mi_residual<<<grid_for(long(rows)*MI_HIDDEN),256,0,stream>>>(h,o->proj,r0,size_t(rows)*MI_HIDDEN)));
    }
    if(cb && ((i+1)%5==0)) {
      MI_LAUNCH((cast_float<<<(size_t(n)*MI_HIDDEN+255)/256,256,0,stream>>>(h,o->x,size_t(n)*MI_HIDDEN)));
      MI_RUN(observe(o,cb,ctx,i+1,o->x,n,MI_HIDDEN,false));
    }
  }
  MI_RUN(cuteafd_vision_rms_h2048(o->attention,h,bf(s.merger_norm),o->norm,n,stream));
  // The residual stream is dead: its extent (n x 2048 BF16 = units x 8192) holds the unfold.
  MI_LAUNCH((mi_unfold<<<grid_for(long(units)*MI_MERGED),256,0,stream>>>(o->norm,h,gh,gw)));
  uint16_t* staged=reinterpret_cast<uint16_t*>(o->x);   // merged LM rows before the span write
  for(int u0=0;u0<units;u0+=CHUNK) {
    int rows=std::min(CHUNK,units-u0);
    MI_RUN(gemm(o,h+size_t(u0)*MI_MERGED,s.merger_extra[1],rows,MI_HIDDEN,MI_MERGED,o->proj));
    MI_LAUNCH((mi_cast<<<grid_for(long(rows)*MI_HIDDEN),256,0,stream>>>(o->proj,o->mlp,size_t(rows)*MI_HIDDEN)));
    MI_RUN(gemm(o,o->mlp,s.merger_fc1,rows,MI_LM,MI_HIDDEN,o->fc1));
    MI_LAUNCH((mi_gelu<<<grid_for(long(rows)*MI_LM),256,0,stream>>>(o->fc1,o->gelu,size_t(rows)*MI_LM)));
    MI_RUN(gemm(o,o->gelu,s.merger_fc2,rows,MI_LM,MI_LM,o->fc2));
    MI_LAUNCH((mi_cast<<<grid_for(long(rows)*MI_LM),256,0,stream>>>(o->fc2,staged+size_t(u0)*MI_LM,size_t(rows)*MI_LM)));
  }
  MI_LAUNCH((mi_span<<<grid_for(long(uh)*(uw+1)*MI_LM),256,0,stream>>>(staged,bf(s.merger_extra[4]),
    bf(s.merger_extra[5]),o->output,uh,uw)));
  MI_RUN(observe(o,cb,ctx,31,o->output,uh*(uw+1),MI_LM,false));
  MI_RUN(cudaMemcpyAsync(output,o->output,output_bytes,cudaMemcpyDeviceToHost,stream));
  MI_RUN(cudaStreamSynchronize(stream));++o->ledger.encodes;
#undef MI_LAUNCH
#undef MI_RUN
  return 0;
}
