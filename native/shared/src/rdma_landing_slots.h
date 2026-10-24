#pragma once

#include <cstddef>

namespace cuteafd {

// WQEs consume landing slots in successful post order, independently of the
// host header ring's depth. A failed post must retry the same landing slot.
struct RdmaLandingSlots {
  size_t bytes = 0;
  size_t count = 0;
  size_t next = 0;

  size_t offset() const { return next * bytes; }
  void posted() { next = (next + 1) % count; }
};

}  // namespace cuteafd
