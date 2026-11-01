// Official Muse layouts; all transformer arithmetic is dispatched through b12x.
constexpr int MUSE_HIDDEN=1536, MUSE_PATCH=1176, MUSE_INTER=8960;

size_t scratch_muse(Owner* o,size_t tokens,int output_width) {
  size_t cursor=0,n=tokens*4;
  auto take=[&](auto& ptr,size_t bytes) {
    using P=typename std::remove_reference<decltype(ptr)>::type;
    ptr=o->arena ? reinterpret_cast<P>(o->arena+cursor) : nullptr;
    cursor+=aligned(bytes);
  };
  take(o->muse_x,n*1536*2);take(o->muse_xt,n*1536*2);take(o->norm,n*1536*2);
  take(o->patches,n*1176*2);take(o->muse_qkv,n*4608*2);
  take(o->q,n*1536*2);take(o->k,n*1536*2);take(o->v,n*1536*2);
  take(o->attn,n*1536*2);take(o->muse_proj,n*1536*2);take(o->mlp,n*8960*2);
  take(o->muse_adapter,tokens*4096*2);take(o->output,tokens*output_width*2);
  take(o->muse_cos,n*96*2);take(o->muse_sin,n*96*2);
  take(o->rgb,n*14*14*3);take(o->lut,3*256*4);
  take(o->col,n*4);take(o->inverse,n*4);
  take(o->pos_indices,n*4*4);take(o->pos_weights,n*4*4);
  take(o->cu_seqlens,513*4);take(o->lse,size_t(16384)*16*4);
  take(o->workspace,BLAS_BYTES);
  return cursor;
}

bool valid_muse(const cuteafd_vision_spec& s) {
  if(s.abi_version!=2 || s.reserved!=5 || s.geometry_reserved || !s.max_tokens || s.max_tokens>4096 ||
     s.hidden!=1536 || s.depth!=50 || s.heads!=16 || s.kv_heads!=16 || s.head_dim!=96 ||
     s.intermediate!=8960 || s.patch_size!=14 || s.merger_width!=6144 || s.output_width!=8192 ||
     s.norm_eps!=1e-5f || !s.weight_bytes || s.weight_bytes>(4ULL<<30) ||
     s.patch_bias!=UINT64_MAX || s.merger_fc1_bias!=UINT64_MAX || s.merger_fc2_bias!=UINT64_MAX)return false;
  const uint64_t offsets[]={s.patch,s.pos_embed,s.merger_extra[0],s.merger_extra[1],s.merger_norm,
    s.merger_norm_bias,s.merger_fc1,s.merger_fc2,s.merger_extra[2],s.inv_freq};
  const size_t sizes[]={1536*1176*2,1024*1536*2,1536*2,1536*2,1536*2,1536*2,
    4096*6144*2,4096*4096*2,8192*4096*2,24*4};
  for(int i=0;i<10;++i)if(!extent(s,offsets[i],sizes[i]))return false;
  for(int i=3;i<8;++i)if(s.merger_extra[i]!=UINT64_MAX)return false;
  for(int i=0;i<50;++i) {
    const auto& b=s.blocks[i];
    const uint64_t off[]={b.qkv,b.qkv_bias,b.proj,b.proj_bias,b.gate_up,b.gate_up_bias,
      b.down,b.down_bias,b.norm1,b.norm2,s.norm1_bias[i],s.norm2_bias[i]};
    const size_t bytes[]={4608*1536*2,4608*2,1536*1536*2,1536*2,8960*1536*2,8960*2,
      1536*8960*2,1536*2,1536*2,1536*2,1536*2,1536*2};
    for(int j=0;j<12;++j)if(!extent(s,off[j],bytes[j]))return false;
    if((b.window!=0 && b.window!=32) || b.column_order || b.key0_bias!=UINT64_MAX ||
       s.q_norm[i]!=UINT64_MAX || s.k_norm[i]!=UINT64_MAX)return false;
  }
  return true;
}

__global__ void muse_rgb_patches(const uint8_t* rgb,const float* lut,uint16_t* out,int gh,int gw) {
  size_t total=size_t(gh)*gw*1176;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x) {
    int p=i/1176,d=i%588,c=d/196,pixel=d%196;
    int y=(p/gw)*14+pixel/14,x=(p%gw)*14+pixel%14;
    out[i]=bf16_bits(lut[c*256+rgb[(size_t(y)*gw*14+x)*3+c]]);
  }
}

// Match the official swapped-grid half-pixel interpolation and permuted +1 RoPE.
__global__ void muse_position_metadata(int32_t* indices,float* weights,uint16_t* cosines,
  uint16_t* sines,const int32_t* permutation,const float* freq,int gh,int gw) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=gh*gw)return;
  float sy=__fsub_rn(__fdiv_rn(__fmul_rn(float(row/gh)+.5f,32.f),float(gw)),.5f);
  float sx=__fsub_rn(__fdiv_rn(__fmul_rn(float(row%gh)+.5f,32.f),float(gh)),.5f);
  int fy=int(floorf(sy)),fx=int(floorf(sx));
  for(int y=0;y<2;++y)for(int x=0;x<2;++x) {
    int iy=fy+y,ix=fx+x,tap=y*2+x;
    indices[row*4+tap]=max(0,min(31,iy))*32+max(0,min(31,ix));
    float wy=fmaxf(0.f,1.f-fabsf(__fsub_rn(__fsub_rn(sy,float(fy)),float(y))));
    float wx=fmaxf(0.f,1.f-fabsf(__fsub_rn(__fsub_rn(sx,float(fx)),float(x))));
    weights[row*4+tap]=(iy>=0 && iy<32 && ix>=0 && ix<32) ? __fmul_rn(wy,wx) : 0.f;
  }
  int p=permutation[row];
  for(int d=0;d<96;++d) {
    int phase=d%48,position=phase<24 ? p%gw+1 : p/gw+1;
    float angle=__fmul_rn(float(position),freq[phase%24]);
    cosines[size_t(row)*96+d]=bf16_bits(cosf(angle));
    sines[size_t(row)*96+d]=bf16_bits(sinf(angle));
  }
}

#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
int muse_program(Owner* o,const char* name,std::initializer_list<void*> pointers,
  std::initializer_list<uint64_t> scalars) {
  return cuteafd_vision_pointwise_launch(o->attention,name,pointers.begin(),scalars.begin(),o->stream);
}
int muse_norm(Owner* o,uint16_t* x,uint64_t weight,uint64_t bias,uint16_t* out,int rows) {
  return muse_program(o,"vision_layer_norm_h1536",{x,o->weights+weight,o->weights+bias,
    out,out,out,out},{uint64_t(rows)});
}
int muse_gather(Owner* o,uint16_t* x,int32_t* indices,uint16_t* out,int rows) {
  return muse_program(o,"vision_row_gather_h1536",{x,indices,out},{uint64_t(rows),1,0});
}
#endif

int encode_muse(Owner* o,const uint8_t* rgb,uint64_t rgb_bytes,const float* lut,int gh,int gw,
  uint16_t* output,uint64_t output_bytes,cuteafd_vision_observer cb,void* ctx) {
#ifndef CUTEAFD_HAVE_VISION_ATTENTION_AOT
  return cudaErrorNotSupported;
#else
  const auto& s=o->spec;int64_t count=int64_t(gh)*gw;
  if(!o->uploaded || !rgb || !lut || !output || gh<2 || gw<2 || gh%2 || gw%2 ||
     count>int64_t(s.max_tokens)*4 || rgb_bytes!=uint64_t(count)*14*14*3 ||
     output_bytes!=uint64_t(count/4)*8192*2)return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  int n=int(count),units=n/4;auto stream=o->stream;
  auto& permutation=o->muse_permutation;auto& inverse=o->muse_inverse;auto& windows=o->muse_windows;
  int cursor=0,segments=0;windows[0]=0;
  for(int y0=0;y0<gh;y0+=32)for(int x0=0;x0<gw;x0+=32) {
    for(int y=y0;y<std::min(gh,y0+32);++y)for(int x=x0;x<std::min(gw,x0+32);++x) {
      int p=y*gw+x;permutation[cursor]=p;inverse[p]=cursor++;
    }
    windows[++segments]=cursor;
  }
  std::fill(windows.begin()+segments+1,windows.end(),n);
#define MUSE_RUN(expr) do { int rc=int(expr);if(rc)return drain(o,rc); } while(0)
#define MUSE_LAUNCH(expr) do { expr;MUSE_RUN(cudaGetLastError()); } while(0)
  MUSE_RUN(cudaMemcpyAsync(o->rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  MUSE_RUN(cudaMemcpyAsync(o->lut,lut,768*4,cudaMemcpyHostToDevice,stream));
  MUSE_RUN(cudaMemcpyAsync(o->col,permutation.data(),size_t(n)*4,cudaMemcpyHostToDevice,stream));
  MUSE_RUN(cudaMemcpyAsync(o->inverse,inverse.data(),size_t(n)*4,cudaMemcpyHostToDevice,stream));
  MUSE_RUN(cudaMemcpyAsync(o->cu_seqlens,windows.data(),513*4,cudaMemcpyHostToDevice,stream));
  MUSE_LAUNCH((muse_rgb_patches<<<(size_t(n)*1176+255)/256,256,0,stream>>>(o->rgb,o->lut,o->patches,gh,gw)));
  MUSE_LAUNCH((muse_position_metadata<<<(n+255)/256,256,0,stream>>>(o->pos_indices,o->pos_weights,
    o->muse_cos,o->muse_sin,o->col,reinterpret_cast<const float*>(o->weights+s.inv_freq),gh,gw)));
  MUSE_RUN(observe(o,cb,ctx,-1,o->patches,n,1176,false));
  MUSE_RUN(observe(o,cb,ctx,-2,o->muse_cos,n,96,false));
  MUSE_RUN(observe(o,cb,ctx,-3,o->muse_sin,n,96,false));
  MUSE_RUN(muse_program(o,"vision_projection_h1176_o1536",{o->patches,o->weights+s.patch,o->muse_x},{uint64_t(n)}));
  MUSE_RUN(muse_program(o,"vision_position_gather_h1536",{o->weights+s.pos_embed,o->pos_indices,
    o->pos_weights,o->muse_x,o->muse_xt},{uint64_t(n)}));
  MUSE_RUN(observe(o,cb,ctx,0,o->muse_xt,n,1536,false));
  MUSE_RUN(muse_norm(o,o->muse_xt,s.merger_extra[0],s.merger_extra[1],o->norm,n));
  MUSE_RUN(muse_gather(o,o->norm,o->col,o->muse_x,n));
  MUSE_RUN(observe(o,cb,ctx,51,o->muse_x,n,1536,false));
  auto* x=o->muse_x;auto* xt=o->muse_xt;
  for(int i=0;i<50;++i) {
    const auto& b=s.blocks[i];
    MUSE_RUN(muse_norm(o,x,b.norm1,s.norm1_bias[i],o->norm,n));
    MUSE_RUN(muse_program(o,"vision_biased_projection_h1536_o4608",
      {o->norm,o->weights+b.qkv,o->weights+b.qkv_bias,o->muse_qkv},{uint64_t(n)}));
    for(int part=0;part<3;++part) {
      auto* target=part==0 ? o->q : part==1 ? o->k : o->v;
      MUSE_RUN(cudaMemcpy2DAsync(target,1536*2,o->muse_qkv+part*1536,4608*2,1536*2,n,cudaMemcpyDeviceToDevice,stream));
    }
    MUSE_RUN(muse_program(o,"vision_axial_rotary_d96_f32",
      {o->q,o->k,o->muse_cos,o->muse_sin,o->norm,o->muse_proj},{uint64_t(n)}));
    if(!b.window)MUSE_LAUNCH((vision_lengths<<<1,1,0,stream>>>(o->cu_seqlens,n)));
    else MUSE_RUN(cudaMemcpyAsync(o->cu_seqlens,windows.data(),513*4,cudaMemcpyHostToDevice,stream));
    MUSE_RUN(cuteafd_vision_attention_launch(b.window ? o->window_attention : o->attention,
      o->norm,o->muse_proj,o->v,o->attn,o->lse,o->cu_seqlens,1.f/sqrtf(96.f),stream));
    MUSE_RUN(muse_program(o,"vision_biased_projection_h1536_o1536",
      {o->attn,o->weights+b.proj,o->weights+b.proj_bias,o->muse_proj},{uint64_t(n)}));
    MUSE_RUN(muse_program(o,"vision_residual_h1536",{x,o->muse_proj,xt},{uint64_t(n)}));
    std::swap(x,xt);
    MUSE_RUN(muse_norm(o,x,b.norm2,s.norm2_bias[i],o->norm,n));
    MUSE_RUN(muse_program(o,"vision_biased_projection_h1536_o8960",
      {o->norm,o->weights+b.gate_up,o->weights+b.gate_up_bias,o->mlp},{uint64_t(n)}));
    MUSE_RUN(muse_program(o,"vision_gelu_h8960",{o->mlp,o->mlp},{uint64_t(n)}));
    MUSE_RUN(muse_program(o,"vision_biased_projection_h8960_o1536",
      {o->mlp,o->weights+b.down,o->weights+b.down_bias,o->muse_proj},{uint64_t(n)}));
    MUSE_RUN(muse_program(o,"vision_residual_h1536",{x,o->muse_proj,xt},{uint64_t(n)}));
    std::swap(x,xt);
    MUSE_RUN(observe(o,cb,ctx,i+1,x,n,1536,false));
  }
  MUSE_RUN(muse_gather(o,x,o->inverse,xt,n));
  MUSE_RUN(observe(o,cb,ctx,52,xt,n,1536,false));
  MUSE_RUN(muse_norm(o,xt,s.merger_norm,s.merger_norm_bias,o->norm,n));
  MUSE_RUN(observe(o,cb,ctx,53,o->norm,n,1536,false));
  MUSE_RUN(muse_program(o,"vision_spatial_merge_h1536_r2",{o->norm,o->muse_qkv},{uint64_t(units),uint64_t(gh),uint64_t(gw)}));
  MUSE_RUN(observe(o,cb,ctx,54,o->muse_qkv,units,6144,false));
  MUSE_RUN(muse_program(o,"vision_projection_h6144_o4096",{o->muse_qkv,o->weights+s.merger_fc1,o->muse_adapter},{uint64_t(units)}));
  MUSE_RUN(muse_program(o,"vision_gelu_h4096",{o->muse_adapter,o->muse_adapter},{uint64_t(units)}));
  MUSE_RUN(muse_program(o,"vision_projection_h4096_o4096",{o->muse_adapter,o->weights+s.merger_fc2,o->muse_qkv},{uint64_t(units)}));
  MUSE_RUN(muse_program(o,"vision_gelu_h4096",{o->muse_qkv,o->muse_qkv},{uint64_t(units)}));
  MUSE_RUN(observe(o,cb,ctx,55,o->muse_qkv,units,4096,false));
  MUSE_RUN(muse_program(o,"vision_projection_h4096_o8192",{o->muse_qkv,o->weights+s.merger_extra[2],o->output},{uint64_t(units)}));
  MUSE_RUN(observe(o,cb,ctx,56,o->output,units,8192,false));
  MUSE_RUN(cudaMemcpyAsync(output,o->output,output_bytes,cudaMemcpyDeviceToHost,stream));
  MUSE_RUN(cudaStreamSynchronize(stream));++o->ledger.encodes;
#undef MUSE_LAUNCH
#undef MUSE_RUN
  return 0;
#endif
}
