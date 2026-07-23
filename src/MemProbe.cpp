// Copyright the Hypnos authors.
// SPDX-License-Identifier: Apache-2.0
#include "MemProbe.h"

#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// CMake already defines NOMINMAX for this target; guard against C4005.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <cstdlib>
#endif

namespace hyp {
namespace {

// Apple Silicon has no discrete GPU memory to query (see ProbeMemory), so the
// NVIDIA probes and the loader they need are compiled out entirely there.
#if !defined(__APPLE__)

// ---------------------------------------------------------------------------
// Tiny cross-platform dynamic-library helper. We never link these libraries; a
// missing one simply means "no GPU information available", not an error.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
using LibHandle = HMODULE;
LibHandle OpenLib(const char* name) { return LoadLibraryA(name); }
void* GetSym(LibHandle h, const char* name) {
  return h ? reinterpret_cast<void*>(GetProcAddress(h, name)) : nullptr;
}
void CloseLib(LibHandle h) { if (h) FreeLibrary(h); }
#else
using LibHandle = void*;
LibHandle OpenLib(const char* name) { return dlopen(name, RTLD_LAZY | RTLD_LOCAL); }
void* GetSym(LibHandle h, const char* name) { return h ? dlsym(h, name) : nullptr; }
void CloseLib(LibHandle h) { if (h) dlclose(h); }
#endif

// ---------------------------------------------------------------------------
// NVML. Preferred over cudart because querying it does NOT create a CUDA context
// (a context costs hundreds of MB of the very VRAM we are trying to measure).
// ---------------------------------------------------------------------------
struct NvmlMemory {  // layout of nvmlMemory_t (v1): total, free, used
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

const char* const kNvmlNames[] = {
#if defined(_WIN32)
    "nvml.dll",
    // Older drivers only install NVML under the NVSMI directory.
    "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll",
#elif defined(__linux__)
    "libnvidia-ml.so.1",
    "libnvidia-ml.so",
#endif
    nullptr,
};

bool ProbeNvml(int device_id, uint64_t* free_bytes, uint64_t* total_bytes) {
  LibHandle lib = nullptr;
  for (int i = 0; kNvmlNames[i] != nullptr && lib == nullptr; ++i)
    lib = OpenLib(kNvmlNames[i]);
  if (!lib) return false;

  using InitFn = int (*)(void);
  using ShutdownFn = int (*)(void);
  using HandleFn = int (*)(unsigned int, void**);
  using MemFn = int (*)(void*, NvmlMemory*);

  auto nvmlInit = reinterpret_cast<InitFn>(GetSym(lib, "nvmlInit_v2"));
  if (!nvmlInit) nvmlInit = reinterpret_cast<InitFn>(GetSym(lib, "nvmlInit"));
  auto nvmlShutdown = reinterpret_cast<ShutdownFn>(GetSym(lib, "nvmlShutdown"));
  auto nvmlHandle =
      reinterpret_cast<HandleFn>(GetSym(lib, "nvmlDeviceGetHandleByIndex_v2"));
  if (!nvmlHandle)
    nvmlHandle = reinterpret_cast<HandleFn>(GetSym(lib, "nvmlDeviceGetHandleByIndex"));
  auto nvmlMem = reinterpret_cast<MemFn>(GetSym(lib, "nvmlDeviceGetMemoryInfo"));

  bool ok = false;
  if (nvmlInit && nvmlHandle && nvmlMem && nvmlInit() == 0) {
    void* dev = nullptr;
    NvmlMemory mem{};
    // NOTE: NVML enumerates by PCI order, which need not match the CUDA ordinal
    // when CUDA_VISIBLE_DEVICES is set. We use device 0 to match OrtAccel.h's
    // CUDA device_id 0; on a multi-GPU box with reordering this can read the
    // wrong card, which the headroom margin and the OOM retry ladder absorb.
    if (nvmlHandle(static_cast<unsigned int>(device_id), &dev) == 0 && dev != nullptr &&
        nvmlMem(dev, &mem) == 0 && mem.total > 0) {
      *free_bytes = mem.free;
      *total_bytes = mem.total;
      ok = true;
    }
    if (nvmlShutdown) nvmlShutdown();
  }
  CloseLib(lib);
  return ok;
}

// ---------------------------------------------------------------------------
// cudart fallback. Only used when NVML is absent; cudaMemGetInfo implicitly
// initialises a context on the current device, so it perturbs what it measures.
// ---------------------------------------------------------------------------
const char* const kCudartNames[] = {
#if defined(_WIN32)
    "cudart64_12.dll", "cudart64_110.dll",
#elif defined(__linux__)
    "libcudart.so.12", "libcudart.so.11.0", "libcudart.so",
#endif
    nullptr,
};

bool ProbeCudart(uint64_t* free_bytes, uint64_t* total_bytes) {
  LibHandle lib = nullptr;
  for (int i = 0; kCudartNames[i] != nullptr && lib == nullptr; ++i)
    lib = OpenLib(kCudartNames[i]);
  if (!lib) return false;

  using MemGetInfoFn = int (*)(size_t*, size_t*);
  auto cudaMemGetInfo = reinterpret_cast<MemGetInfoFn>(GetSym(lib, "cudaMemGetInfo"));
  bool ok = false;
  if (cudaMemGetInfo) {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) == 0 && t > 0) {
      *free_bytes = f;
      *total_bytes = t;
      ok = true;
    }
  }
  CloseLib(lib);
  return ok;
}

#endif  // !__APPLE__

// ---------------------------------------------------------------------------
// Host RAM.
// ---------------------------------------------------------------------------
void ProbeSystem(uint64_t* avail, uint64_t* total) {
  *avail = 0;
  *total = 0;
#if defined(_WIN32)
  MEMORYSTATUSEX st{};
  st.dwLength = sizeof(st);
  if (GlobalMemoryStatusEx(&st)) {
    *total = st.ullTotalPhys;
    *avail = st.ullAvailPhys;
  }
#elif defined(__APPLE__)
  uint64_t memsize = 0;
  size_t len = sizeof(memsize);
  if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) *total = memsize;

  // Free + inactive + purgeable approximates what can be allocated without
  // pressure. On Apple Silicon this is also the "VRAM" budget, since the GPU
  // shares this pool — there is no separate device memory to query.
  vm_size_t page = 0;
  mach_port_t host = mach_host_self();
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_page_size(host, &page) == KERN_SUCCESS &&
      host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm),
                        &count) == KERN_SUCCESS) {
    *avail = (static_cast<uint64_t>(vm.free_count) +
              static_cast<uint64_t>(vm.inactive_count) +
              static_cast<uint64_t>(vm.purgeable_count)) *
             static_cast<uint64_t>(page);
  }
#elif defined(__linux__)
  // MemAvailable is the kernel's own estimate of allocatable-without-swapping,
  // which is what we want; MemFree alone badly understates it because of cache.
  if (FILE* f = std::fopen("/proc/meminfo", "r")) {
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
      unsigned long long kb = 0;
      if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) *total = kb * 1024ULL;
      else if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) *avail = kb * 1024ULL;
    }
    std::fclose(f);
  }
  if (*total == 0) {
    long pages = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGESIZE);
    if (pages > 0 && ps > 0) *total = static_cast<uint64_t>(pages) * ps;
  }
#endif
  if (*avail == 0) *avail = *total;  // better an overestimate than a zero budget
}

}  // namespace

MemInfo ProbeMemory(int device_id) {
  MemInfo m;
  ProbeSystem(&m.sys_avail, &m.sys_total);

#if defined(__APPLE__)
  // Apple Silicon has unified memory: the CoreML EP draws from the same pool as
  // the CPU and exposes no device-memory API, so the system figures ARE the GPU
  // budget. Reporting gpu_valid=false keeps callers on the system-memory path
  // rather than inventing a device budget that does not exist.
  (void)device_id;
#else
  if (ProbeNvml(device_id, &m.gpu_free, &m.gpu_total)) {
    m.gpu_valid = true;
    m.gpu_source = "NVML";
  } else if (ProbeCudart(&m.gpu_free, &m.gpu_total)) {
    m.gpu_valid = true;
    m.gpu_source = "cudart";
  }
#endif
  return m;
}

std::string FormatBytes(uint64_t bytes) {
  char buf[64];
  const double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  if (gb >= 1.0) {
    std::snprintf(buf, sizeof(buf), "%.1f GB", gb);
  } else {
    std::snprintf(buf, sizeof(buf), "%.0f MB",
                  static_cast<double>(bytes) / (1024.0 * 1024.0));
  }
  return std::string(buf);
}

std::string DescribeMemory(const MemInfo& m) {
  std::string s;
  if (m.gpu_valid) {
    s += "GPU " + FormatBytes(m.gpu_free) + "/" + FormatBytes(m.gpu_total) + " free";
    if (!m.gpu_source.empty()) s += " (" + m.gpu_source + ")";
    s += ", ";
  }
  s += "RAM " + FormatBytes(m.sys_avail) + "/" + FormatBytes(m.sys_total) + " available";
  return s;
}

}  // namespace hyp
