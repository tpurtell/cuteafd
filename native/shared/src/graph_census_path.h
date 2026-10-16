#pragma once
// Where the graph census appends its per-capture records (one JSON line each).
// With a profile directory (CUTEAFD_PROFILE_DIR, mounted at the same path on
// the host) the file lands there, named by host like the other profile
// writers, so the launch's profile collection copies it back. Without one it
// stays container-local.
#include <string>

inline std::string cuteafd_graph_census_captures_path(const char* directory, const char* host) {
  if (directory == nullptr || directory[0] != '/') {
    return "/tmp/cuteafd-graph-census-captures.jsonl";
  }
  std::string name;
  for (const char* p = host != nullptr ? host : ""; *p != '\0'; ++p) {
    const char c = *p;
    const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                      c == '-' || c == '_';
    name += safe ? c : '_';
  }
  if (name.empty()) { name = "unknown"; }
  std::string path(directory);
  while (path.size() > 1 && path.back() == '/') { path.pop_back(); }
  if (path != "/") { path += '/'; }
  return path + "graph-census-captures-" + name + ".jsonl";
}
