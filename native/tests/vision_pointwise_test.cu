// Exercise the production Qwen pointwise kernels beyond grid_for's block cap.
#include "../shared/cuda/vision_runtime.cu"
#include <cstdio>
#include <vector>

struct DeviceBuffer {
  void* ptr=nullptr;
  ~DeviceBuffer() { if(ptr)cudaFree(ptr); }
};
#define CHECK(expr) do { int rc=int(expr);if(rc) { std::fprintf(stderr,"%s: %d\n",#expr,rc);return 1; } } while(0)
int main() {
  // Registration must own every host array before Rust/ctypes releases it.
  for(uint32_t depth : {24u,27u,28u,30u,50u}) {
    Owner owner;
    {
      std::vector<cuteafd_vision_block> blocks(depth);
      std::vector<uint64_t> n1(depth,11),n2(depth,22),q(depth,33),k(depth,44);
      blocks.back().norm1=1234;
      cuteafd_vision_spec spec{};
      spec.storage_abi=3;spec.depth=depth;spec.blocks=blocks.data();
      spec.norm1_bias=n1.data();spec.norm2_bias=n2.data();
      spec.q_norm=q.data();spec.k_norm=k.data();
      if(!copy_spec(owner,spec) || owner.spec.blocks==spec.blocks ||
         owner.spec.q_norm==spec.q_norm)return 1;
      blocks.back().norm1=0;q.back()=0;
    }
    if(owner.spec.depth!=depth || owner.spec.blocks[depth-1].norm1!=1234 ||
       owner.spec.norm1_bias[depth-1]!=11 || owner.spec.norm2_bias[depth-1]!=22 ||
       owner.spec.q_norm[depth-1]!=33 || owner.spec.k_norm[depth-1]!=44)return 1;
  }
  int devices=0;
  if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
  CHECK(cudaSetDevice(0));
  constexpr size_t count=4096*256+1, width=1152, rows=(count+width-1)/width;
  DeviceBuffer x,y,bias,table,indices,weights,out;
  CHECK(cudaMalloc(&x.ptr,count*sizeof(float)));
  CHECK(cudaMalloc(&y.ptr,count*sizeof(float)));
  CHECK(cudaMalloc(&bias.ptr,width*sizeof(float)));
  CHECK(cudaMalloc(&table.ptr,width*sizeof(uint16_t)));
  CHECK(cudaMalloc(&indices.ptr,rows*4*sizeof(int32_t)));
  CHECK(cudaMalloc(&weights.ptr,rows*4*sizeof(float)));
  CHECK(cudaMalloc(&out.ptr,count*sizeof(uint16_t)));
  std::vector<float> ones(count,1.f), twos(width,2.f), taps(rows*4,0.f), actual(count);
  for(size_t row=0;row<rows;++row)taps[row*4]=1.f;
  std::vector<uint16_t> table_values(width,0x3f80), bits(count);
  CHECK(cudaMemcpy(y.ptr,ones.data(),count*4,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(bias.ptr,twos.data(),width*4,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(table.ptr,table_values.data(),width*2,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(indices.ptr,0,rows*4*4));
  CHECK(cudaMemcpy(weights.ptr,taps.data(),rows*4*4,cudaMemcpyHostToDevice));
  for(int mode=0;mode<5;++mode) {
    CHECK(cudaMemcpy(x.ptr,ones.data(),count*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out.ptr,0xff,count*2));
    if(mode==0)qwen_patch_position<<<grid_for(count),256>>>(static_cast<float*>(x.ptr),static_cast<float*>(bias.ptr),
      static_cast<uint16_t*>(table.ptr),static_cast<int32_t*>(indices.ptr),static_cast<float*>(weights.ptr),count);
    if(mode==1)qwen_residual<<<grid_for(count),256>>>(static_cast<float*>(x.ptr),static_cast<float*>(y.ptr),static_cast<float*>(bias.ptr),count);
    if(mode==2 || mode==3)qwen_biased_gelu<<<grid_for(count),256>>>(static_cast<float*>(x.ptr),static_cast<float*>(bias.ptr),
      static_cast<uint16_t*>(out.ptr),count,width,mode==2);
    if(mode==4)qwen_biased_cast<<<grid_for(count),256>>>(static_cast<float*>(x.ptr),static_cast<float*>(bias.ptr),
      static_cast<uint16_t*>(out.ptr),count,width);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    if(mode<2) {
      CHECK(cudaMemcpy(actual.data(),x.ptr,count*4,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<count;++i)if(actual[i]!=4.f) {
        std::fprintf(stderr,"mode %d element %zu was not written: %g\n",mode,i,actual[i]);return 1;
      }
    } else {
      CHECK(cudaMemcpy(bits.data(),out.ptr,count*2,cudaMemcpyDeviceToHost));
      float expected=mode==4?3.f:mode==2?1.5f*(1.f+tanhf(0.7978845608028654f*(3.f+0.044715f*27.f))):1.5f*(1.f+erff(3.f*0.7071067811865475f));
      __nv_bfloat16 rounded=__float2bfloat16_rn(expected);
      uint16_t want=*reinterpret_cast<uint16_t*>(&rounded);
      for(size_t i=0;i<count;++i)if(bits[i]!=want) {
        std::fprintf(stderr,"mode %d element %zu was not written: %04x, expected %04x\n",mode,i,bits[i],want);return 1;
      }
    }
  }
  CHECK(cudaMemset(x.ptr,0xff,count*4));
  qwen_cast_float<<<grid_for(count),256>>>(static_cast<uint16_t*>(out.ptr),static_cast<float*>(x.ptr),count);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(actual.data(),x.ptr,count*4,cudaMemcpyDeviceToHost));
  for(size_t i=0;i<count;++i)if(actual[i]!=3.f) {
    std::fprintf(stderr,"observer cast element %zu was not written: %g\n",i,actual[i]);return 1;
  }
  std::printf("Qwen capped-launch pointwise coverage passed for all %zu elements\n",count);
  return 0;
}
