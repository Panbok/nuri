#pragma once

#if defined(_WIN32)
  #if defined(NURI_SHARED)
    #if defined(NURI_EXPORTS)
      #define NURI_API __declspec(dllexport)
    #else
      #define NURI_API __declspec(dllimport)
    #endif
  #else
    #define NURI_API
  #endif
#else
  #if defined(NURI_SHARED)
    #define NURI_API __attribute__((visibility("default")))
  #else
    #define NURI_API
  #endif
#endif