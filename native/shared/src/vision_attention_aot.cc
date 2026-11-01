#include "cuteafd_vision_attention_internal.h"
#include "vision_owner_entries.h"
#include <cuda_runtime.h>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

namespace {
using ModuleFn = void (*)(void**);
using LaunchFn = void (*)(void**, int32_t);
constexpr int MAX_DEVICES = 32, MAX_ARGUMENTS = 32;
struct Program {
  const char* name;
  ModuleFn initialize;
  ModuleFn load;
  LaunchFn launch;
  uint32_t pointers;
  const char* scalars;
  cudaLibrary_t library = nullptr;
  bool loaded[MAX_DEVICES]{};
};
struct Entry {
  int head_dim;
  bool windowed;
  const char* pointwise_group;
  std::vector<Program> programs;
  uint32_t references = 0;
};
Entry entries[] = {CUTEAFD_VISION_OWNER_ENTRIES};
std::mutex mutex;
struct Attention { Entry* entry; int device; };

Entry* find(int dim, bool windowed, const char* pointwise) {
  for(auto& entry:entries)
    if(entry.head_dim==dim && entry.windowed==windowed && !std::strcmp(entry.pointwise_group,pointwise))return &entry;
  return nullptr;
}
int release(Entry& entry) {
  int first=0;
  for(auto& program:entry.programs) {
    if(!program.library)continue;
    int status=cudaLibraryUnload(program.library);
    if(!first)first=status;
    // Keep failed unloads addressable so a later owner can retry cleanup.
    if(!status) { program.library=nullptr;for(auto& flag:program.loaded)flag=false; }
  }
  return first;
}
int load(Entry& entry,int device) {
  for(auto& program:entry.programs) {
    int result=0;
    auto* library=&program.library;
    if(!program.library) {
      void* args[]={&library,&result};program.initialize(args);
    }
    if(!result && !program.loaded[device]) {
      void* args[]={&library,&device,&result};program.load(args);
      if(!result)program.loaded[device]=true;
    }
    if(result) {
      // A failed first owner must not leak modules; existing owners stay live.
      if(!entry.references)release(entry);
      return result;
    }
  }
  return 0;
}
int launch(Attention* owner,Program& program,void* const* pointers,const uint64_t* scalars,void* stream) {
  const size_t n=std::strlen(program.scalars);
  if((program.pointers && !pointers) || (n && !scalars) || program.pointers+n+2>MAX_ARGUMENTS)
    return cudaErrorInvalidValue;
  int device=-1,status=cudaGetDevice(&device);if(status)return status;
  if(device!=owner->device)return cudaErrorInvalidDevice;
  if(!program.loaded[device])return cudaErrorInvalidResourceHandle;
  void* values[MAX_ARGUMENTS];void* args[MAX_ARGUMENTS];
  int32_t i32[MAX_ARGUMENTS];int64_t i64[MAX_ARGUMENTS];float f32[MAX_ARGUMENTS];
  uint32_t count=0;
  for(uint32_t i=0;i<program.pointers;++i) { values[i]=pointers[i];args[count++]=&values[i]; }
  for(size_t i=0;i<n;++i) {
    switch(program.scalars[i]) {
      case 'i':i32[i]=static_cast<int32_t>(scalars[i]);args[count++]=&i32[i];break;
      case 'l':i64[i]=static_cast<int64_t>(scalars[i]);args[count++]=&i64[i];break;
      case 'f': { uint32_t bits=static_cast<uint32_t>(scalars[i]);std::memcpy(&f32[i],&bits,4);args[count++]=&f32[i];break; }
      default:return cudaErrorInvalidValue;
    }
  }
  int result=0;args[count++]=&stream;args[count++]=&result;
  program.launch(args,static_cast<int32_t>(count));return result;
}
}

extern "C" int32_t cuteafd_vision_attention_create_entry(int32_t dim,int32_t windowed,const char* pointwise,void** out) {
  if(!out)return cudaErrorInvalidValue;
  *out=nullptr;
  if(!pointwise || (windowed!=0 && windowed!=1))return cudaErrorInvalidValue;
  auto* entry=find(dim,windowed!=0,pointwise);if(!entry)return cudaErrorInvalidValue;
  int device=-1,status=cudaGetDevice(&device);if(status)return status;
  if(device<0 || device>=MAX_DEVICES)return cudaErrorInvalidDevice;
  auto* owner=new(std::nothrow)Attention{entry,device};if(!owner)return cudaErrorMemoryAllocation;
  std::lock_guard<std::mutex> guard(mutex);
  status=load(*entry,device);
  if(status) { delete owner;return status; }
  ++entry->references;*out=owner;return 0;
}
extern "C" int32_t cuteafd_vision_attention_destroy(void* opaque) {
  if(!opaque)return 0;
  auto* owner=static_cast<Attention*>(opaque);
  // The caller drains its stream before releasing this entry's modules.
  std::lock_guard<std::mutex> guard(mutex);
  int result=--owner->entry->references ? 0 : release(*owner->entry);
  delete owner;return result;
}
extern "C" int32_t cuteafd_vision_attention_launch(void* opaque,void* q,void* k,void* v,
    void* out,void* lse,void* cu,float scale,void* stream) {
  if(!opaque)return cudaErrorInitializationError;
  auto* owner=static_cast<Attention*>(opaque);auto& program=owner->entry->programs.front();
  // Legacy exports pass separate q/k lengths and an unused sink; shared
  // varlen exports take the same live boundaries through one pointer.
  if((program.pointers!=8 && program.pointers!=6) || std::strcmp(program.scalars,"f"))
    return cudaErrorInvalidValue;
  void* pointers[]={q,k,v,out,lse,cu,cu,nullptr};
  uint32_t bits;std::memcpy(&bits,&scale,4);uint64_t scalar=bits;
  return launch(owner,program,pointers,&scalar,stream);
}
extern "C" int32_t cuteafd_vision_pointwise_launch(void* opaque,const char* name,void* const* pointers,
    const uint64_t* scalars,void* stream) {
  if(!opaque)return cudaErrorInitializationError;
  if(!name)return cudaErrorInvalidValue;
  auto* owner=static_cast<Attention*>(opaque);
  for(size_t i=1;i<owner->entry->programs.size();++i) {
    auto& program=owner->entry->programs[i];
    if(!std::strcmp(name,program.name))return launch(owner,program,pointers,scalars,stream);
  }
  return cudaErrorInvalidValue;
}
