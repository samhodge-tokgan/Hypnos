// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
//
// Runtime probing of available GPU (VRAM) and system memory, used to size the
// matting inference so it cannot OOM the accelerator or the host.
//
// Everything here is resolved with dlopen/LoadLibrary at CALL time: there is no
// build-time or link-time dependency on CUDA or NVML, so the plugin still loads
// and runs (on CPU) on a machine with no NVIDIA driver installed.
#pragma once

#include <cstdint>
#include <string>

namespace hyp {

struct MemInfo {
  // GPU (only meaningful when gpu_valid). Bytes.
  bool gpu_valid = false;
  uint64_t gpu_free = 0;
  uint64_t gpu_total = 0;
  // Host RAM. Bytes. sys_avail is "can be allocated without swapping", which is
  // what matters for the CPU fallback path; it is not simply total - used.
  uint64_t sys_avail = 0;
  uint64_t sys_total = 0;
  // Where the GPU numbers came from: "NVML", "cudart", or "" when unavailable.
  std::string gpu_source;
};

// Probe the current memory state. `device_id` indexes the NVML/CUDA device and
// should match the execution provider's device (we use 0, as OrtAccel.h does).
//
// Cheap enough to call per render, but the result is a snapshot: another process
// (or the host itself) can allocate between the probe and our session creation,
// which is why callers must ALSO keep a headroom margin and handle the OOM
// exception path. See MatteEngine's retry ladder.
MemInfo ProbeMemory(int device_id = 0);

// Human-readable one-liner for status params / messages, e.g.
// "GPU 18.4/24.0 GB free (NVML), RAM 41.2/64.0 GB available".
std::string DescribeMemory(const MemInfo& m);

// Format a byte count as a short human string ("1.5 GB").
std::string FormatBytes(uint64_t bytes);

}  // namespace hyp
