cmake_minimum_required(VERSION 3.20)
project(MEMatteOFX VERSION 0.2.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(HYP_UNIVERSAL "Build a universal arm64+x86_64 binary (macOS)" ON)
option(HYP_WITH_ONNX "Link ONNX Runtime and enable matting inference" ON)
set(HYP_ORT_VERSION "1.27.1" CACHE STRING "ONNX Runtime version to fetch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Windows: link the C++ runtime STATICALLY into the .ofx.
#
# An OFX plugin is loaded into whatever host the user runs, and hosts ship their
# own Visual C++ runtime in their application directory, which wins on the loader
# path. Nuke 16.1v3 ships MSVCP140.dll 14.36 (System32 here has 14.44); a plugin
# that dynamically imports MSVCP140 gets the HOST's version, and if it needs a
# symbol only present in a newer one it fails to load. That exact mismatch is why
# the bundled ONNX Runtime is a static-CRT build (see the ORT_URL comment below);
# measured on this machine the .ofx imported 38 MSVCP140 symbols, so it carried
# the same latent risk.
#
# Static linking removes the dependency entirely. It is safe here because the OFX
# boundary is a C ABI: no CRT objects (FILE*, allocations, C++ types) cross
# between plugin and host, so separate heaps cannot cause a problem.
if(MSVC)
  # CMP0091 (NEW by default at cmake_minimum_required >= 3.15) makes this
  # property the way runtime selection works, rather than editing CMAKE_CXX_FLAGS.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# GCC < 9 (e.g. Rocky Linux 8's default 8.5) keeps std::filesystem in a separate
# library that must be linked explicitly. Empty (a no-op link) on newer toolchains.
set(HYP_FS_LIB "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  set(HYP_FS_LIB stdc++fs)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The prebuilt ONNX Runtime with the CoreML EP is arm64-only, so an ONNX-enabled
# build is arm64-only (which is also what the Apple Neural Engine path requires).
if(APPLE AND HYP_WITH_ONNX)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "arm64 (ONNX Runtime CoreML)" FORCE)
elseif(APPLE AND HYP_UNIVERSAL)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "OFX host arch must match" FORCE)
endif()

# ---------------------------------------------------------------------------
# Vendor the OpenFX SDK (populate only; we compile the Support library ourselves
# rather than adding OpenFX's own CMake targets). Pointing SOURCE_SUBDIR at a
# directory with no CMakeLists.txt makes MakeAvailable download without building.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(openfx
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
  GIT_TAG OFX_Release_1.5.1
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR include            # no CMakeLists here -> populate only
)
FetchContent_MakeAvailable(openfx)

set(OFX_ROOT "${openfx_SOURCE_DIR}")
set(OFX_SUPPORT "${OFX_ROOT}/Support")

# ---------------------------------------------------------------------------
# ONNX Runtime (prebuilt binary release: headers + shared library).
#   macOS arm64 -> CoreML EP ; Linux/Windows x64 -> CUDA EP (GPU) + CPU.
# Populate-only (not a CMake project).
# ---------------------------------------------------------------------------
set(ORT_BASE "https://github.com/microsoft/onnxruntime/releases/download/v${HYP_ORT_VERSION}")
if(HYP_WITH_ONNX)
  set(ORT_HASH_ARG "")
  if(APPLE)
    set(ORT_URL "${ORT_BASE}/onnxruntime-osx-arm64-${HYP_ORT_VERSION}.tgz")
    set(ORT_HASH_ARG URL_HASH SHA256=e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6)
    set(ORT_SHLIB "lib/libonnxruntime.${HYP_ORT_VERSION}.dylib")
    set(HYP_ACCEL "CoreML")
  elseif(WIN32)
    # Windows uses a STATIC-CRT ONNX Runtime built from source (NOT Microsoft's prebuilt),
    # so the bundled runtime + providers carry no external MSVCP140/VCRUNTIME140 dependency
    # and are immune to whatever VC++ runtime the host forces process-wide. (Nuke 16.1 ships
    # MSVCP140 14.36 in its app dir; prebuilt ORT 1.27 needs >=14.40 and fails to init with
    # LoadLibrary error 1114.) Reused verbatim from the humbaba project's pinned release asset.
    set(ORT_URL "https://github.com/samhodge-tokgan/humbaba/releases/download/ort-static-win-${HYP_ORT_VERSION}/onnxruntime-win-x64-static-cuda9arch-${HYP_ORT_VERSION}.zip")
    set(ORT_HASH_ARG URL_HASH SHA256=76a9f67f3704486cc354824212492fcdc152bbe6801667140335aa9aa14026d4)
    set(ORT_SHLIB "lib/onnxruntime.dll")          # runtime (IMPORTED_LOCATION)
    set(ORT_IMPLIB "lib/onnxruntime.lib")         # import lib (IMPORTED_IMPLIB)
    set(HYP_ACCEL "CUDA")
  elseif(UNIX)  # Linux
    set(ORT_URL "${ORT_BASE}/onnxruntime-linux-x64-gpu_cuda12-${HYP_ORT_VERSION}.tgz")
    # (hash added once pinned; FetchContent fetches without it meanwhile)
    set(ORT_SHLIB "lib/libonnxruntime.so.${HYP_ORT_VERSION}")
    set(HYP_ACCEL "CUDA")
  else()
    message(FATAL_ERROR "Unsupported platform for HYP_WITH_ONNX")
  endif()

  FetchContent_Declare(onnxruntime URL "${ORT_URL}" ${ORT_HASH_ARG})
  FetchContent_MakeAvailable(onnxruntime)
  set(ORT_DIR "${onnxruntime_SOURCE_DIR}")
  set(ORT_DYLIB "${ORT_DIR}/${ORT_SHLIB}")

  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DYLIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
  )
  if(WIN32)
    # On Windows link against the import library; IMPORTED_LOCATION is the DLL.
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_IMPLIB "${ORT_DIR}/${ORT_IMPLIB}")
  endif()
  message(STATUS "ONNX Runtime accelerator EP: ${HYP_ACCEL}")
endif()

# ---------------------------------------------------------------------------
# OpenFX C++ Support library
# ---------------------------------------------------------------------------
add_library(ofxsupport STATIC
  ${OFX_SUPPORT}/Library/ofxsCore.cpp
  ${OFX_SUPPORT}/Library/ofxsImageEffect.cpp
  ${OFX_SUPPORT}/Library/ofxsInteract.cpp
  ${OFX_SUPPORT}/Library/ofxsLog.cpp
  ${OFX_SUPPORT}/Library/ofxsMultiThread.cpp
  ${OFX_SUPPORT}/Library/ofxsParams.cpp
  ${OFX_SUPPORT}/Library/ofxsProperty.cpp
  ${OFX_SUPPORT}/Library/ofxsPropertyValidation.cpp
)
target_include_directories(ofxsupport PUBLIC
  ${OFX_ROOT}/include
  ${OFX_SUPPORT}/include
  ${OFX_SUPPORT}/Plugins/include
)
target_compile_definitions(ofxsupport PUBLIC OFX_SUPPORTS_MULTITHREAD)
set_target_properties(ofxsupport PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Per-platform .ofx.bundle layout (OFX packaging spec).
# ---------------------------------------------------------------------------
# Isolation: the bundled ONNX Runtime is given a PRIVATE name so the dynamic loader
# can never deduplicate it against a host's or sibling plugin's copy of the same
# library (which would bind us to a different, ABI-incompatible ORT and crash).
# Note the name differs from humbaba's (_da3) so the two bundles can coexist in one
# host process with independent ORT copies.
# OFX_LIB_ORIG is the name the .ofx links against; OFX_LIB_BUNDLED is the private
# name we rename it to (and repoint the .ofx + providers at) post-build.
if(APPLE)
  set(OFX_ARCH_DIR "MacOS")        # Contents/MacOS/*.ofx
  set(OFX_LIB_DIR "Frameworks")    # bundled shared libs
  set(OFX_RPATH "@loader_path/../Frameworks")
  set(OFX_LIB_ORIG "libonnxruntime.1.dylib")
  set(OFX_LIB_BUNDLED "libonnxruntime_hyp.1.dylib")
elseif(WIN32)
  set(OFX_ARCH_DIR "Win64")        # Contents/Win64/*.ofx (OFX-spec arch dir)
  set(OFX_LIB_DIR "Win64")         # DLLs alongside the .ofx
  set(OFX_RPATH "")                # Windows has no rpath; DLLs found via loader dir
  set(OFX_LIB_ORIG "onnxruntime.dll")
  # The .ofx delay-imports "onnxruntime.dll"; the WinLoader delay hook redirects that
  # import to this PRIVATE name. Without it, Windows ships its own onnxruntime.dll in
  # System32 (Windows ML, a DIFFERENT ORT version) — hosts like Nuke make it resident,
  # and the loader's base-name dedup then binds us to the OS copy instead of ours.
  # Must match kPrivateOrtDll in src/WinLoader.cpp.
  set(OFX_LIB_BUNDLED "onnxruntime_hyp.dll")
else()                             # Linux
  # OFX packaging spec: the 64-bit Linux arch dir is "Linux-x86-64" (hyphens).
  # Using an underscore makes OFX hosts (Natron) silently skip the binary.
  set(OFX_ARCH_DIR "Linux-x86-64") # Contents/Linux-x86-64/*.ofx
  set(OFX_LIB_DIR "Linux-x86-64")  # .so alongside the .ofx
  set(OFX_RPATH "$ORIGIN")
  set(OFX_LIB_ORIG "libonnxruntime.so.1")
  set(OFX_LIB_BUNDLED "libonnxruntime_hyp.so.1")
endif()

# ---------------------------------------------------------------------------
# The plugin: a loadable module named MEMatte.ofx
# ---------------------------------------------------------------------------
add_library(MEMatte MODULE src/MEMattePlugin.cpp)
target_link_libraries(MEMatte PRIVATE ofxsupport ${HYP_FS_LIB})
set_target_properties(MEMatte PROPERTIES
  PREFIX ""
  OUTPUT_NAME "MEMatte"
  SUFFIX ".ofx"
)
if(APPLE)
  # OFX plugins are Mach-O bundles.
  target_link_options(MEMatte PRIVATE -bundle)
endif()

# Export ONLY the OFX C entry points, so none of our C++ / static-lib symbols leak
# into the host process and interpose on the host or a sibling OFX plugin.
# (Windows already exports only the __declspec(dllexport) OFX entry points.)
if(NOT WIN32)
  target_compile_options(MEMatte PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif()
if(APPLE)
  target_link_options(MEMatte PRIVATE
    "-Wl,-exported_symbols_list,${CMAKE_SOURCE_DIR}/packaging/ofx_exports.macos.sym")
elseif(UNIX)
  target_link_options(MEMatte PRIVATE
    "-Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/ofx_exports.map")
endif()

if(HYP_WITH_ONNX)
  target_sources(MEMatte PRIVATE
    src/MatteEngine.cpp
    src/MemProbe.cpp
    src/Sizing.cpp
    src/Trimap.cpp
  )
  if(WIN32)
    # onnxruntime.dll and its dependents live next to the .ofx (Contents/Win64),
    # which the loader does NOT search by default for a LoadLibrary'd plugin.
    # WinLoader.cpp (DllMain) redirects the delay-loaded import to the bundled,
    # privately-named runtime by full path.
    target_sources(MEMatte PRIVATE src/WinLoader.cpp)
    target_link_libraries(MEMatte PRIVATE delayimp)
    target_link_options(MEMatte PRIVATE /DELAYLOAD:onnxruntime.dll)
    target_compile_definitions(MEMatte PRIVATE _USE_MATH_DEFINES NOMINMAX)
  endif()
  target_link_libraries(MEMatte PRIVATE onnxruntime)
  target_compile_definitions(MEMatte PRIVATE HYP_WITH_ONNX=1)
  # Find the bundled shared library relative to the .ofx binary at runtime.
  if(NOT WIN32)
    set_target_properties(MEMatte PROPERTIES
      BUILD_WITH_INSTALL_RPATH ON
      INSTALL_RPATH "${OFX_RPATH}"
    )
  endif()
endif()

# MemProbe dlopen()s NVML / libcudart at runtime (no build-time CUDA dependency).
if(UNIX AND NOT APPLE)
  target_link_libraries(MEMatte PRIVATE ${CMAKE_DL_LIBS})
endif()

# ---------------------------------------------------------------------------
# Assemble the .ofx.bundle (macOS layout: Contents/{Info.plist,MacOS/*.ofx})
# ---------------------------------------------------------------------------
set(OFX_BUNDLE_NAME "MEMatte")
set(OFX_BUNDLE_BINARY "MEMatte.ofx")
set(OFX_BUNDLE_IDENTIFIER "com.tokgan.openfx.MEMatte")
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle")

add_custom_command(TARGET MEMatte POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:MEMatte>" "${BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}/${OFX_BUNDLE_BINARY}"
  COMMENT "Assembling ${OFX_BUNDLE_NAME}.ofx.bundle (${OFX_ARCH_DIR})"
  VERBATIM
)

# Info.plist is a macOS bundle requirement (Linux bundles don't use it).
if(APPLE)
  configure_file("${CMAKE_SOURCE_DIR}/packaging/Info.plist.in" "${CMAKE_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET MEMatte POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/Info.plist" "${BUNDLE_DIR}/Contents/Info.plist"
    VERBATIM)
endif()

# Bundle the ONNX Runtime shared library inside the .ofx.bundle so the plugin is
# self-contained. The plugin links against the SONAME (@rpath/libonnxruntime.1.dylib
# on macOS, libonnxruntime.so.1 on Linux), so the bundled file must carry that name
# (the tarball's real file is fully versioned).
if(HYP_WITH_ONNX)
  add_custom_command(TARGET MEMatte POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "${ORT_DYLIB}"
            "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/${OFX_LIB_BUNDLED}"
    COMMENT "Bundling ONNX Runtime shared library"
    VERBATIM
  )
  # The CUDA EP lives in separate provider shared libraries (cuda/tensorrt/shared);
  # bundle them alongside the .ofx so the EP loads. (Linux: .so, Windows: .dll.)
  if(WIN32)
    file(GLOB ORT_PROVIDER_LIBS "${ORT_DIR}/lib/onnxruntime_providers_*.dll")
  elseif(NOT APPLE)
    file(GLOB ORT_PROVIDER_LIBS "${ORT_DIR}/lib/libonnxruntime_providers_*.so")
  endif()
  if(NOT APPLE)
    foreach(pl ${ORT_PROVIDER_LIBS})
      add_custom_command(TARGET MEMatte POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${pl}" "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/"
        VERBATIM)
    endforeach()
  endif()

  # Rename the bundled ONNX Runtime to a private name and repoint the .ofx (and, on
  # Linux, the provider libs that NEED it) at that name, so the loader can never
  # deduplicate it against a host's / sibling plugin's copy.
  if(NOT "${OFX_LIB_ORIG}" STREQUAL "${OFX_LIB_BUNDLED}")
    set(_bundled "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/${OFX_LIB_BUNDLED}")
    set(_ofxbin "${BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}/${OFX_BUNDLE_BINARY}")
    if(APPLE)
      add_custom_command(TARGET MEMatte POST_BUILD
        COMMAND install_name_tool -id "@rpath/${OFX_LIB_BUNDLED}" "${_bundled}"
        COMMAND install_name_tool -change "@rpath/${OFX_LIB_ORIG}" "@rpath/${OFX_LIB_BUNDLED}" "${_ofxbin}"
        COMMAND codesign --force --sign - "${_bundled}"
        COMMAND codesign --force --sign - "${_ofxbin}"
        COMMENT "Isolating ONNX Runtime (private install name ${OFX_LIB_BUNDLED})"
        VERBATIM)
    elseif(UNIX)
      find_program(PATCHELF patchelf)
      if(NOT PATCHELF)
        message(FATAL_ERROR "patchelf not found (needed to privately rename the bundled "
          "ONNX Runtime). Install it: dnf install patchelf  /  apt-get install patchelf")
      endif()
      add_custom_command(TARGET MEMatte POST_BUILD
        COMMAND ${PATCHELF} --set-soname "${OFX_LIB_BUNDLED}" "${_bundled}"
        COMMAND ${PATCHELF} --replace-needed "${OFX_LIB_ORIG}" "${OFX_LIB_BUNDLED}" "${_ofxbin}"
        COMMENT "Isolating ONNX Runtime (private soname ${OFX_LIB_BUNDLED})"
        VERBATIM)
      foreach(pl ${ORT_PROVIDER_LIBS})
        get_filename_component(_pln "${pl}" NAME)
        add_custom_command(TARGET MEMatte POST_BUILD
          COMMAND ${PATCHELF} --replace-needed "${OFX_LIB_ORIG}" "${OFX_LIB_BUNDLED}"
                  "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/${_pln}"
          VERBATIM)
      endforeach()
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Optionally bundle the ONNX models into Contents/Resources (release build).
# Each variant is a backbone + decoder pair; configure with e.g.
#   -DHYP_MODEL_DIR=/path/to/models   (containing mematte_{s,b}_{backbone,decoder}.onnx)
# ---------------------------------------------------------------------------
if(HYP_WITH_ONNX AND HYP_MODEL_DIR)
  foreach(v s b)
    foreach(part backbone decoder)
      set(_m "${HYP_MODEL_DIR}/mematte_${v}_${part}.onnx")
      if(EXISTS "${_m}")
        add_custom_command(TARGET MEMatte POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/Resources"
          COMMAND ${CMAKE_COMMAND} -E copy "${_m}" "${BUNDLE_DIR}/Contents/Resources/"
          COMMENT "Bundling mematte_${v}_${part}.onnx into Resources"
          VERBATIM)
      endif()
    endforeach()
  endforeach()
endif()

# Ship the model-fetch scripts inside the bundle (model-less installers pull the
# ONNX files post-install, keeping release assets under the size limit).
foreach(f fetch_models.sh fetch_models.ps1)
  if(EXISTS "${CMAKE_SOURCE_DIR}/tools/${f}")
    add_custom_command(TARGET MEMatte POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/tools/${f}"
              "${BUNDLE_DIR}/Contents/${f}"
      VERBATIM)
  endif()
endforeach()

# ---------------------------------------------------------------------------
# Provider-check tool: prints the ONNX Runtime execution providers and probes the
# available GPU/system memory. Built only with ONNX enabled.
# ---------------------------------------------------------------------------
if(HYP_WITH_ONNX)
  add_executable(ort_check tests/ort_check.cpp src/MemProbe.cpp)
  target_include_directories(ort_check PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(ort_check PRIVATE onnxruntime ${HYP_FS_LIB})
  if(UNIX AND NOT APPLE)
    target_link_libraries(ort_check PRIVATE ${CMAKE_DL_LIBS})
  endif()
  if(WIN32)
    # No rpath on Windows: copy the ORT DLLs next to ort_check.exe so it can run.
    file(GLOB ORT_ALL_DLLS "${ORT_DIR}/lib/*.dll")
    foreach(dll ${ORT_ALL_DLLS})
      add_custom_command(TARGET ort_check POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:ort_check>"
        VERBATIM)
    endforeach()
  else()
    set_target_properties(ort_check PROPERTIES
      BUILD_RPATH "${ORT_DIR}/lib"
      INSTALL_RPATH "${ORT_DIR}/lib"
    )
  endif()
endif()

# ---------------------------------------------------------------------------
# Unit tests for the pure-logic pieces (no ONNX Runtime, no OFX host needed):
# the trimap decoder and the memory-sizing model. `ctest` or `make test`.
# ---------------------------------------------------------------------------
enable_testing()
add_executable(trimap_test tests/trimap_test.cpp src/Trimap.cpp)
target_include_directories(trimap_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME trimap COMMAND trimap_test)

add_executable(sizing_test tests/sizing_test.cpp src/Sizing.cpp src/MemProbe.cpp)
target_include_directories(sizing_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
if(UNIX AND NOT APPLE)
  target_link_libraries(sizing_test PRIVATE ${CMAKE_DL_LIBS})
endif()
add_test(NAME sizing COMMAND sizing_test)

add_executable(oom_test tests/oom_test.cpp)
target_include_directories(oom_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME oom COMMAND oom_test)

# ---------------------------------------------------------------------------
# Install: drop the bundle into the user's OFX plugin dir (override with
# -DOFX_INSTALL_DIR=...). Use `cmake --install build` or the `install-local` target.
# ---------------------------------------------------------------------------
if(NOT DEFINED OFX_INSTALL_DIR)
  if(APPLE)
    set(OFX_INSTALL_DIR "$ENV{HOME}/Library/OFX/Plugins")
  elseif(WIN32)
    set(OFX_INSTALL_DIR "$ENV{CommonProgramFiles}/OFX/Plugins")
  else()  # Linux
    set(OFX_INSTALL_DIR "$ENV{HOME}/OFX/Plugins")
  endif()
endif()
install(DIRECTORY "${BUNDLE_DIR}" DESTINATION "${OFX_INSTALL_DIR}" USE_SOURCE_PERMISSIONS)

add_custom_target(install-local
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_INSTALL_DIR}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${OFX_INSTALL_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${BUNDLE_DIR}" "${OFX_INSTALL_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle"
  DEPENDS MEMatte
  COMMENT "Installing bundle to ${OFX_INSTALL_DIR}"
)
