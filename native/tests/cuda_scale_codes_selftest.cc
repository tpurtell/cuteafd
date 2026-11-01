// Load-time scale coding from the same mapped-host sources/jobs used by push
// and mount loaders. The independent CPU oracle also reconstructs every byte.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuteafd_experts.h"
#include "picojson.h"

namespace {
using Job = cuteafd_scale_code_job_t;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void check(cudaError_t status, const char* operation) {
  require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}

template <class T>
struct Mapped {
  T* data = nullptr;
  explicit Mapped(size_t count) {
    check(cudaHostAlloc(reinterpret_cast<void**>(&data), count * sizeof(T), cudaHostAllocMapped),
          "cudaHostAlloc");
  }
  ~Mapped() { cudaFreeHost(data); }
  Mapped(const Mapped&) = delete;
  Mapped& operator=(const Mapped&) = delete;
};
struct Plane {
  uint8_t* data = nullptr;
  size_t bytes;
  static constexpr size_t guard = 64;
  explicit Plane(size_t n) : bytes(n) {
    check(cudaMalloc(reinterpret_cast<void**>(&data), bytes + guard), "cudaMalloc");
    check(cudaMemset(data, 0, bytes), "zero generated plane");
    check(cudaMemset(data + bytes, 0xa5, guard), "initialize guard");
  }
  ~Plane() { cudaFree(data); }
  Plane(const Plane&) = delete;
  Plane& operator=(const Plane&) = delete;
  void compare(const std::vector<uint8_t>& expected, const std::string& name) const {
    std::vector<uint8_t> actual(bytes + guard);
    check(cudaMemcpy(actual.data(), data, actual.size(), cudaMemcpyDeviceToHost), "read coded plane");
    require(expected.size() == bytes, name + ": CPU extent mismatch");
    for (size_t i = 0; i < bytes; ++i)
      require(actual[i] == expected[i], name + ": coded byte mismatch at " + std::to_string(i));
    require(std::all_of(actual.begin() + bytes, actual.end(), [](uint8_t x) { return x == 0xa5; }),
            name + ": wrote beyond admitted plane");
  }
};

size_t plane_bytes(size_t rows, size_t columns) { return rows * (2 * ((columns + 3) / 4) + 1); }

// False means the first unsupported source row, as in the Rust CPU packer.
bool cpu_pack(const Job& job, std::vector<uint8_t>& out, uint32_t* refused = nullptr) {
  const size_t tiles = (job.columns + 3) / 4;
  const size_t code_bytes = job.stored_rows * tiles * 2;
  for (uint32_t row = 0; row < job.rows; ++row) {
    const uint8_t* source = job.source + row * job.pitch;
    const auto limits = std::minmax_element(source, source + job.columns);
    if (unsigned(*limits.second) - *limits.first > 15) {
      if (refused) *refused = row;
      return false;
    }
    const size_t resident = job.start_row + row;
    out[code_bytes + resident] = *limits.first;
    for (size_t column = 0; column < job.columns; ++column) {
      const size_t start = resident / 256 * 256;
      const size_t tile_rows = std::min<size_t>(256, job.stored_rows - start);
      const size_t word = start * tiles + column / 4 * tile_rows + resident % 256;
      const unsigned shift = unsigned(column % 4) * 4;
      out[word * 2 + shift / 8] |= uint8_t((source[column] - *limits.first) << (shift % 8));
    }
  }
  // Decode logical bytes independently, including a partial final K128 group.
  for (size_t row = 0; row < job.rows; ++row) {
    const size_t resident = job.start_row + row;
    for (size_t column = 0; column < tiles * 4; ++column) {
      const size_t start = resident / 256 * 256;
      const size_t tile_rows = std::min<size_t>(256, job.stored_rows - start);
      const size_t word = start * tiles + column / 4 * tile_rows + resident % 256;
      const uint8_t offset = (out[word * 2 + column % 4 / 2] >> (column % 2 * 4)) & 15;
      if (column < job.columns)
        require(unsigned(offset) + out[code_bytes + resident] == job.source[row * job.pitch + column],
                "CPU oracle did not reconstruct the checkpoint byte");
      else
        require(offset == 0, "CPU oracle has nonzero K-tail offsets");
    }
  }
  return true;
}

void launch(const std::vector<Job>& source, uint64_t expected_error = UINT64_MAX) {
  Mapped<Job> jobs(source.size());
  Mapped<uint64_t> error(1);
  std::copy(source.begin(), source.end(), jobs.data);
  *error.data = UINT64_MAX;
  check(cudaError_t(cuteafd_v41_pack_scale_codes_async(jobs.data, uint32_t(source.size()), error.data,
                                                     nullptr)), "pack scale codes");
  check(cudaDeviceSynchronize(), "drain scale codes");
  require(*error.data == expected_error, "GPU refusal job/row mismatch");
}

picojson::value parse(const std::string& text) {
  picojson::value value;
  const auto error = picojson::parse(value, text);
  require(error.empty(), "invalid checkpoint JSON: " + error);
  return value;
}

struct Tensor {
  std::string name;
  size_t rows, columns;
  std::vector<uint8_t> bytes;
};
Tensor read_tensor(const std::filesystem::path& snapshot, const picojson::value& index,
                   const std::string& name) {
  const auto& map = index.get("weight_map").get<picojson::object>();
  const auto found = map.find(name);
  require(found != map.end(), "checkpoint has no " + name);
  std::ifstream shard(snapshot / found->second.get<std::string>(), std::ios::binary);
  require(bool(shard), "cannot open tensor shard for " + name);
  uint64_t header_bytes = 0;
  shard.read(reinterpret_cast<char*>(&header_bytes), sizeof(header_bytes));
  require(bool(shard) && header_bytes < (1ull << 28), "invalid safetensors header");
  std::string header(header_bytes, '\0');
  shard.read(header.data(), header.size());
  require(bool(shard), "short safetensors header");
  const auto metadata = parse(header).get(name);
  require(metadata.get("dtype").get<std::string>() == "F8_E8M0", name + ": expected UE8M0");
  const auto& shape = metadata.get("shape").get<picojson::array>();
  const auto& offsets = metadata.get("data_offsets").get<picojson::array>();
  require(shape.size() == 2 && offsets.size() == 2, name + ": invalid scale shape");
  Tensor tensor{name, size_t(shape[0].get<double>()), size_t(shape[1].get<double>()), {}};
  const auto start = uint64_t(offsets[0].get<double>());
  const auto end = uint64_t(offsets[1].get<double>());
  require(tensor.rows && tensor.columns && end >= start && end - start == tensor.rows * tensor.columns,
          name + ": invalid scale extent");
  tensor.bytes.resize(end - start);
  shard.seekg(8 + header_bytes + start);
  shard.read(reinterpret_cast<char*>(tensor.bytes.data()), tensor.bytes.size());
  require(bool(shard), name + ": short scale read");
  return tensor;
}

void real_expert(const std::filesystem::path& snapshot, const picojson::value& index,
                 const std::string& prefix) {
  auto up = read_tensor(snapshot, index, prefix + ".w3.scale");
  auto gate = read_tensor(snapshot, index, prefix + ".w1.scale");
  auto down = read_tensor(snapshot, index, prefix + ".w2.scale");
  require(up.rows == gate.rows && up.columns == gate.columns && down.rows == up.columns * 32
          && down.columns * 32 == up.rows, prefix + ": incompatible expert geometry");
  Mapped<uint8_t> pinned(up.bytes.size() + gate.bytes.size() + down.bytes.size());
  std::copy(up.bytes.begin(), up.bytes.end(), pinned.data);
  std::copy(gate.bytes.begin(), gate.bytes.end(), pinned.data + up.bytes.size());
  std::copy(down.bytes.begin(), down.bytes.end(), pinned.data + up.bytes.size() + gate.bytes.size());
  require(up.rows % 32 == 0, prefix + ": invalid K32 geometry");
  for (size_t world = 1; world <= 8; ++world) {
    const size_t units = up.rows / 32, base = units / world, extra = units % world;
    for (size_t layer = 0; layer < world; ++layer) {
     for (size_t rank = 0; rank < world; ++rank) {
      const size_t canonical = extra ? (rank + world - layer) % world : rank;
      const size_t width = 32 * (base + size_t(canonical < extra));
      const size_t first = 32 * (canonical * base + std::min(canonical, extra));
      const size_t stored = width;
      Plane s13(plane_bytes(2 * stored, up.columns));
      Plane s2(plane_bytes(down.rows, width / 32));
      const size_t up_start = first * up.columns;
      const size_t down_start = first / 32;
      std::vector<Job> jobs{
          {pinned.data + up_start, s13.data, up.bytes.size() - up_start, s13.bytes,
           up.columns, uint32_t(width), uint32_t(up.columns), uint32_t(2 * stored), 0},
          {pinned.data + up.bytes.size() + up_start, s13.data, gate.bytes.size() - up_start, s13.bytes,
           gate.columns, uint32_t(width), uint32_t(gate.columns), uint32_t(2 * stored), uint32_t(stored)},
          {pinned.data + up.bytes.size() + gate.bytes.size() + down_start, s2.data,
           down.bytes.size() - down_start, s2.bytes, down.columns,
           uint32_t(down.rows), uint32_t(width / 32), uint32_t(down.rows), 0}};
      std::vector<uint8_t> expected13(s13.bytes), expected2(s2.bytes);
      require(cpu_pack(jobs[0], expected13) && cpu_pack(jobs[1], expected13)
              && cpu_pack(jobs[2], expected2), prefix + ": source span exceeds 15");
      launch(jobs);
      const auto label = prefix + " TP" + std::to_string(world) + " layer" + std::to_string(layer)
          + " rank" + std::to_string(rank);
      s13.compare(expected13, label + " gate/up");
      s2.compare(expected2, label + " down");
     }
    }
  }
  std::printf("%s: all TP1..8 rotated exact coded planes byte-exact\n", prefix.c_str());
}

void edges() {
  constexpr size_t rows = 259, columns = 18, pitch = 23, stored = 288;
  Mapped<uint8_t> source(rows * pitch);
  std::fill_n(source.data, rows * pitch, 0xff);
  for (size_t row = 0; row < rows; ++row)
    for (size_t col = 0; col < columns; ++col)
      source.data[row * pitch + col] = row % 2 ? uint8_t(240 + col % 16) : 255;
  Plane plane(plane_bytes(stored, columns));
  Job job{source.data, plane.data, rows * pitch, plane.bytes, pitch, rows, columns, stored, 17};
  std::vector<uint8_t> expected(plane.bytes);
  require(cpu_pack(job, expected), "span 0/15 CPU pack failed");
  launch({job});
  plane.compare(expected, "synthetic span 0/15, pitched rows, N/K tails");

  // Two refusing jobs: atomicMin must retain the earliest job and its first
  // failing row, even when later rows/jobs run before it.
  source.data[7 * pitch] = 239;
  source.data[8 * pitch] = 239;
  source.data[8 * pitch + 1] = 255;
  Plane rejected(plane.bytes);
  job.destination = rejected.data;
  uint32_t row = UINT32_MAX;
  std::vector<uint8_t> refused(rejected.bytes);
  require(!cpu_pack(job, refused, &row) && row == 7, "CPU span-16 refusal row mismatch");
  launch({job, job}, 7);
  std::vector<uint8_t> actual(rejected.bytes);
  check(cudaMemcpy(actual.data(), rejected.data, actual.size(), cudaMemcpyDeviceToHost), "read refused plane");
  const size_t code_bytes = stored * ((columns + 3) / 4) * 2;
  for (size_t bad : {7, 8}) {
    const size_t resident = job.start_row + bad;
    require(actual[code_bytes + resident] == 0, "refused row wrote its base");
    for (size_t tile = 0; tile < (columns + 3) / 4; ++tile) {
      const size_t start = resident / 256 * 256;
      const size_t tile_rows = std::min<size_t>(256, stored - start);
      const size_t at = (start * ((columns + 3) / 4) + tile * tile_rows + resident % 256) * 2;
      require(actual[at] == 0 && actual[at + 1] == 0, "refused row wrote codes");
    }
  }
  Mapped<Job> jobs(1);
  Mapped<uint64_t> error(1);
  *jobs.data = job;
  jobs.data->source_bytes = columns - 1;
  require(cuteafd_v41_pack_scale_codes_async(jobs.data, 1, error.data, nullptr) == cudaErrorInvalidValue,
          "short source not rejected");
  *jobs.data = job;
  --jobs.data->stored_rows;
  require(cuteafd_v41_pack_scale_codes_async(jobs.data, 1, error.data, nullptr) == cudaErrorInvalidValue,
          "unaligned resident rows not rejected");
  std::puts("synthetic span 0/15 exact; span 16 refused with exact job/row; extent guards intact");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
      if (std::getenv("CUTEAFD_REQUIRE_CUDA")) throw std::runtime_error("CUDA device required");
      std::puts("scale codes selftest: no CUDA device, skipping");
      return 77;
    }
    check(cudaSetDevice(0), "cudaSetDevice");
    const char* snapshot = argc > 1 ? argv[1] : std::getenv("CUTEAFD_PUSH_V41");
    require(snapshot && *snapshot, "pass a V4.1 snapshot or set CUTEAFD_PUSH_V41 (real scales are required)");
    std::ifstream file(std::filesystem::path(snapshot) / "model.safetensors.index.json");
    require(bool(file), "cannot open checkpoint index");
    const auto index = parse(std::string(std::istreambuf_iterator<char>(file), {}));
    edges();
    real_expert(snapshot, index, "layers.3.ffn.experts.0");
    for (int stage = 0; stage < 3; ++stage)
      real_expert(snapshot, index, "mtp." + std::to_string(stage) + ".ffn.experts.0");
    std::puts("scale codes selftest: PASS");
    std::puts("test result: ok. 1 passed; 0 failed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "scale codes selftest: FAIL: %s\n", error.what());
    return 1;
  }
}
