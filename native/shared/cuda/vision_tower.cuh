// Independent still-image extension helpers. Included after Owner and the shared
// extent/alignment helpers; PE and existing tower arithmetic are not modified.
bool generic_extent(const cuteafd_vision_spec& s,uint64_t offset,size_t bytes,size_t alignment=2) {
  return offset!=UINT64_MAX && offset%alignment==0 && offset<=s.weight_bytes && bytes<=s.weight_bytes-offset;
}

bool generic_norm_extent(const cuteafd_vision_spec& s,const cuteafd_vision_tower_norm& n,uint32_t width) {
  if(!n.kind)return n.epsilon==0.f && n.weight==UINT64_MAX && n.bias==UINT64_MAX;
  if((n.kind!=1 && n.kind!=2) || !isfinite(n.epsilon) || n.epsilon<=0.f || !generic_extent(s,n.weight,size_t(width)*2))return false;
  return n.bias==UINT64_MAX || (n.kind==1 && generic_extent(s,n.bias,size_t(width)*2));
}

bool generic_projector_extent(const cuteafd_vision_spec& s,const cuteafd_vision_tower_projector& p) {
  if(!p.kind)return !p.activation && !p.input_width && !p.hidden_width && !p.output_width && !p.reserved &&
    p.fc1==UINT64_MAX && p.fc1_bias==UINT64_MAX && p.fc2==UINT64_MAX && p.fc2_bias==UINT64_MAX;
  if(p.kind!=1 || (p.activation!=1 && p.activation!=2) || p.reserved || !p.input_width || !p.hidden_width || !p.output_width ||
     p.input_width>32768 || p.hidden_width>32768 || p.output_width>32768)return false;
  return generic_extent(s,p.fc1,size_t(p.input_width)*p.hidden_width*2) && generic_extent(s,p.fc2,size_t(p.hidden_width)*p.output_width*2) &&
    (p.fc1_bias==UINT64_MAX || generic_extent(s,p.fc1_bias,size_t(p.hidden_width)*2)) &&
    (p.fc2_bias==UINT64_MAX || generic_extent(s,p.fc2_bias,size_t(p.output_width)*2));
}

bool valid_generic_tower(const cuteafd_vision_spec& s) {
  if(!s.tower || s.perception || s.abi_version!=4 || s.geometry_reserved || !s.weight_bytes || s.weight_bytes>(5ULL<<30) ||
     !s.max_tokens || s.max_tokens>16698 || !s.blocks || !s.norm1_bias || !s.norm2_bias || !s.q_norm || !s.k_norm)return false;
  const auto& t=*s.tower;
  // Hy4-flash: patch-16 raster, separate biased Q/K/V, scale-factor positions,
  // channel-major conv grouping and checkpoint newline rows. Others stay patch 14.
  const bool hy4=t.absolute_kind==2;
  if(hy4!=(t.projection_order==3) || s.patch_size!=(hy4?16u:14u))return false;
  if(hy4 && ((s.head_dim!=72 && s.head_dim!=96) || t.rotary_kind || t.pixel_layout!=1 || t.temporal_patch!=1 || t.pre_norm.kind ||
     t.block_norm_kind!=1 || t.activation!=1 || t.final_norm.kind!=2 || t.projectors[0].activation!=1 ||
     t.post_norm.kind || s.max_tokens>4096 || !generic_extent(s,s.merger_extra[0],size_t(s.output_width)*2)))return false;
  if(!hy4 && s.merger_extra[0]!=UINT64_MAX && s.merger_extra[0]!=0)return false;
  if(t.version!=1 || t.size!=sizeof(t) || (t.block_norm_kind!=1 && t.block_norm_kind!=2) ||
     (t.activation!=1 && t.activation!=2) || !isfinite(t.block_norm_epsilon) || t.block_norm_epsilon<=0.f ||
     !s.hidden || s.hidden>8192 || !s.intermediate || s.intermediate>32768 || !s.depth || s.depth>128 ||
     !s.heads || s.heads>32 || s.kv_heads!=s.heads || (s.head_dim!=72 && s.head_dim!=80 && s.head_dim!=96 && s.head_dim!=128) ||
     uint64_t(s.heads)*s.head_dim!=t.attention_width || t.spatial_merge!=2 ||
     (t.pixel_layout!=1 && t.pixel_layout!=2) ||
     !((t.temporal_patch==1 && t.temporal_input==0) || (t.temporal_patch==2 && t.temporal_input==1)))return false;
  if(!generic_norm_extent(s,t.pre_norm,s.hidden) || !generic_norm_extent(s,t.final_norm,s.hidden) ||
     !generic_norm_extent(s,t.post_norm,s.output_width))return false;
  uint64_t rotated=0;
  for(int axis=0;axis<3;++axis) {
    uint32_t dim=t.rotary_axis_dims[axis];rotated+=dim;
    if(dim%2 || (dim ? !generic_extent(s,t.inv_freq[axis],size_t(dim/2)*4,4) : t.inv_freq[axis]!=UINT64_MAX))return false;
  }
  if(rotated!=t.rotated_dims || rotated+t.passthrough_dims!=s.head_dim || t.rotary_kind>2)return false;
  if((!t.rotary_kind && rotated) || (t.rotary_kind==1 && (!t.rotary_axis_dims[0] || !t.rotary_axis_dims[1] || !t.rotary_axis_dims[2])) ||
     (t.rotary_kind==2 && (t.rotary_axis_dims[0] || !t.rotary_axis_dims[1] || t.rotary_axis_dims[1]!=t.rotary_axis_dims[2] || t.passthrough_dims)))return false;
  if(!t.absolute_kind) {
    if(t.position_height || t.position_width || t.interpolation || t.absolute_position!=UINT64_MAX)return false;
  } else if(t.absolute_kind>2 || !t.position_height || t.position_height>512 || !t.position_width || t.position_width>512 ||
     t.interpolation!=t.absolute_kind ||
     // Hy4's learned table keeps the unused class-token row 0 ahead of the grid.
     !generic_extent(s,t.absolute_position,(size_t(t.position_height)*t.position_width+(hy4?1:0))*s.hidden*2))return false;
  for(const auto& p:t.projectors)if(!generic_projector_extent(s,p))return false;
  const auto& first=t.projectors[0];const auto& second=t.projectors[1];
  if(first.kind!=1)return false;
  if(t.projection_order==1 || t.projection_order==3) {
    if(uint64_t(s.hidden)*4!=first.input_width || first.output_width!=s.output_width || second.kind)return false;
  } else if(t.projection_order==2) {
    if(first.input_width!=s.hidden || second.kind!=1 || uint64_t(first.output_width)*4!=second.input_width || second.output_width!=s.output_width)return false;
  } else return false;
  if(!generic_extent(s,s.patch,size_t(s.hidden)*3*t.temporal_patch*s.patch_size*s.patch_size*2) ||
     (s.patch_bias!=UINT64_MAX && !generic_extent(s,s.patch_bias,size_t(s.hidden)*2)))return false;
  for(uint32_t layer=0;layer<s.depth;++layer) {
    const auto& b=s.blocks[layer];
    // Hy4 Q/K/V are adjacent checkpoint tensors: the same 3x extent, read as three projections.
    if(hy4 && (b.qkv_bias==UINT64_MAX || s.norm1_bias[layer]==UINT64_MAX || s.norm2_bias[layer]==UINT64_MAX))return false;
    const uint64_t offsets[]={b.qkv,b.proj,b.gate_up,b.down,b.norm1,b.norm2};
    const size_t sizes[]={size_t(3)*t.attention_width*s.hidden*2,size_t(t.attention_width)*s.hidden*2,
      size_t(s.intermediate)*s.hidden*2,size_t(s.hidden)*s.intermediate*2,size_t(s.hidden)*2,size_t(s.hidden)*2};
    for(int field=0;field<6;++field)if(!generic_extent(s,offsets[field],sizes[field]))return false;
    const uint64_t biases[]={b.qkv_bias,b.proj_bias,b.gate_up_bias,b.down_bias,s.norm1_bias[layer],s.norm2_bias[layer]};
    const size_t widths[]={size_t(3)*t.attention_width,s.hidden,s.intermediate,s.hidden,s.hidden,s.hidden};
    for(int field=0;field<6;++field)if(biases[field]!=UINT64_MAX && !generic_extent(s,biases[field],widths[field]*2))return false;
    if(t.block_norm_kind==2 && (s.norm1_bias[layer]!=UINT64_MAX || s.norm2_bias[layer]!=UINT64_MAX))return false;
    if(b.window || b.column_order || b.key0_bias!=UINT64_MAX || s.q_norm[layer]!=UINT64_MAX || s.k_norm[layer]!=UINT64_MAX)return false;
  }
  return true;
}

size_t scratch_generic_tower(const cuteafd_vision_spec& s,unsigned char* arena,GenericTowerStorage& out) {
  const auto& t=*s.tower;
  const size_t n=size_t(s.max_tokens)*4,h=s.hidden,a=t.attention_width;
  size_t cursor=0;
  auto take=[&](auto& pointer,size_t bytes) {
    using P=typename std::remove_reference<decltype(pointer)>::type;
    pointer=arena?reinterpret_cast<P>(arena+cursor):nullptr;cursor+=aligned(bytes);
  };
  take(out.x,n*h*2);take(out.next,n*h*2);
  take(out.cosines,n*t.rotated_dims/2*4);take(out.sines,n*t.rotated_dims/2*4);take(out.lengths,2*4);
  const size_t phase_base=cursor;size_t peak=0;
  auto phase=[&]() { peak=std::max(peak,cursor-phase_base);cursor=phase_base; };
  // Patch input dies after projection; absolute positions die after their add.
  take(out.rgb,n*s.patch_size*s.patch_size*3);take(out.lut,768*4);
  take(out.patches,n*3*t.temporal_patch*s.patch_size*s.patch_size*2);phase();
  take(out.position,n*h*2);phase();
  // Norm/QKV can occupy the first fused-QKV region plus unused rotary tail.
  take(out.qkv,n*3*a*2);take(out.qkv_norm,n*h*2);phase();
  // Rotary requires disjoint fused source and Q/K/V destinations. Attention
  // reuses the old fused source for O/LSE after rotary has drained on-stream.
  take(out.qkv,n*3*a*2);take(out.q,n*a*2);take(out.k,n*a*2);take(out.v,n*a*2);phase();
  take(out.attention,n*a*2);take(out.lse,n*s.heads*4);phase();
  // Attention projection writes next directly; MLP reuses the entire phase.
  out.projection=out.next;
  take(out.norm,n*h*2);take(out.mlp,n*s.intermediate*2);phase();
  // Projection/grouping stages own distinct source and destination ranges.
  // Their storage aliases transformer temporaries only after final norm.
  const size_t units=s.max_tokens;
  // MiniMax first projects each patch before grouping. Its expanded source and
  // both first-stage linear buffers are simultaneous until the grouping launch.
  if(t.projection_order==2) {
    take(out.preprojector,n*t.projectors[0].hidden_width*2);
    take(out.preprojected,n*t.projectors[0].output_width*2);phase();
  }
  const size_t projected=t.projection_order==2?t.projectors[0].output_width:h;
  const auto& final_projector=t.projectors[t.projection_order==2?1:0];
  const size_t result_bytes=units*final_projector.output_width*2;
  // Keep the final linear result at the same offset as postnorm's source.
  // Kimi group/input and FC1 activation live after it, disjoint through FC2.
  take(out.projector_result,result_bytes);
  if(t.projection_order==2)out.grouped=out.preprojected; // Already patch-group-major.
  else take(out.grouped,units*4*projected*2);
  take(out.projector,units*final_projector.hidden_width*2);phase();
  take(out.postnorm_input,result_bytes);take(out.output,units*s.output_width*2);phase();
  return phase_base+peak;
}

#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
int generic_program(Owner* owner,const std::string& name,std::initializer_list<void*> pointers,
    std::initializer_list<uint64_t> scalars) {
  return cuteafd_vision_pointwise_launch(owner->attention,name.c_str(),pointers.begin(),scalars.begin(),owner->stream);
}
int generic_norm(Owner* owner,const cuteafd_vision_tower_norm& norm,uint16_t* x,uint16_t* output,int rows,int width) {
  if(!norm.kind)return int(cudaMemcpyAsync(output,x,size_t(rows)*width*2,cudaMemcpyDeviceToDevice,owner->stream));
  uint32_t epsilon;std::memcpy(&epsilon,&norm.epsilon,sizeof(epsilon));
  std::string name="vision_tower_"+std::string(norm.kind==1?"layer_norm":"rms_norm")+"_h"+std::to_string(width);
  return generic_program(owner,name,{x,owner->weights+norm.weight,norm.bias==UINT64_MAX?static_cast<void*>(output):static_cast<void*>(owner->weights+norm.bias),
    output},{uint64_t(rows),epsilon});
}
int generic_projection(Owner* owner,uint16_t* x,uint64_t weight,uint64_t bias,uint16_t* output,int rows,int input,int width) {
  std::string name="vision_"+std::string(bias==UINT64_MAX?"projection":"biased_projection")+
    "_h"+std::to_string(input)+"_o"+std::to_string(width);
  if(bias==UINT64_MAX)return generic_program(owner,name,{x,owner->weights+weight,output},{uint64_t(rows)});
  return generic_program(owner,name,{x,owner->weights+weight,owner->weights+bias,output},{uint64_t(rows)});
}
#endif
