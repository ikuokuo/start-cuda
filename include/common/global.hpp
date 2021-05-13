#pragma once

// os

#ifdef _WIN32
# define IK_OS_WIN
# ifdef _WIN64
#  define IK_OS_WIN64
# else
#  define IK_OS_WIN32
# endif
# if defined(__MINGW32__) || defined(__MINGW64__)
#  define IK_OS_MINGW
#  ifdef __MINGW64__
#   define IK_OS_MINGW64
#  else
#   define IK_OS_MINGW32
#  endif
# elif defined(__CYGWIN__) || defined(__CYGWIN32__)
#  define IK_OS_CYGWIN
# endif
#elif __APPLE__
# include <TargetConditionals.h>
# if TARGET_IPHONE_SIMULATOR
#  define IK_OS_IPHONE
#  define IK_OS_IPHONE_SIMULATOR
# elif TARGET_OS_IPHONE
#  define IK_OS_IPHONE
# elif TARGET_OS_MAC
#  define IK_OS_MAC
# else
#  error "Unknown Apple platform"
# endif
#elif __ANDROID__
# define IK_OS_ANDROID
#elif __linux__
# define IK_OS_LINUX
#elif __unix__
# define IK_OS_UNIX
#elif defined(_POSIX_VERSION)
# define IK_OS_POSIX
#else
# error "Unknown compiler"
#endif

#ifndef IK_OS_SEP
# if defined(IK_OS_WIN) && !defined(IK_OS_MINGW)
#  define IK_OS_SEP '\\'
#  define IK_OS_SEP_STR "\\"
# else
#  define IK_OS_SEP '/'
#  define IK_OS_SEP_STR "/"
# endif
#endif

// export

#if defined(IK_OS_WIN)
# define IK_DECL_EXPORT __declspec(dllexport)
# define IK_DECL_IMPORT __declspec(dllimport)
# define IK_DECL_HIDDEN
#else
# define IK_DECL_EXPORT __attribute__((visibility("default")))
# define IK_DECL_IMPORT __attribute__((visibility("default")))
# define IK_DECL_HIDDEN __attribute__((visibility("hidden")))
#endif

#ifdef IK_EXPORTS
# define IK_API IK_DECL_EXPORT
#else
# define IK_API IK_DECL_IMPORT
#endif

#ifdef __cplusplus
# define IK_EXTERN_C       extern "C"
# define IK_EXTERN_C_BEGIN extern "C" {
# define IK_EXTERN_C_END   }
#else
# define IK_EXTERN_C       /* Nothing */
# define IK_EXTERN_C_BEGIN /* Nothing */
# define IK_EXTERN_C_END   /* Nothing */
#endif

// ns

#ifndef IK_NAMESPACE_BEGIN
# define IK_NAMESPACE_BEGIN(name) namespace name {
#endif
#ifndef IK_NAMESPACE_END
# define IK_NAMESPACE_END(name) }
#endif

#ifndef IK_NAMESPACE
# define IK_NAMESPACE ik
#endif
#ifndef IK_BEGIN_NAMESPACE
# define IK_BEGIN_NAMESPACE IK_NAMESPACE_BEGIN(IK_NAMESPACE)
#endif
#ifndef IK_END_NAMESPACE
# define IK_END_NAMESPACE IK_NAMESPACE_END(IK_NAMESPACE)
#endif
#ifndef IK_USE_NAMESPACE
# define IK_USE_NAMESPACE using namespace ::IK_NAMESPACE;  // NOLINT
#endif

// helper

#ifndef IK_STRINGIFY_HELPER
# define IK_STRINGIFY_HELPER(X) #X
#endif
#ifndef IK_STRINGIFY
# define IK_STRINGIFY(X) IK_STRINGIFY_HELPER(X)
#endif

#ifndef IK_DISABLE_COPY
# define IK_DISABLE_COPY(Class) \
  Class(const Class &) = delete;\
  Class &operator=(const Class &) = delete;
#endif

#ifndef IK_DISABLE_MOVE
# define IK_DISABLE_MOVE(Class) \
  Class(Class &&) = delete; \
  Class &operator=(Class &&) = delete;
#endif

#define IK_DISABLE_COPY_MOVE(Class) \
  IK_DISABLE_COPY(Class) \
  IK_DISABLE_MOVE(Class)

#ifndef IK_UNUSED
# define IK_UNUSED(x) (void)x;
#endif

#ifndef IK_UNKNOWN
# define IK_UNKNOWN "UNKNOWN"
#endif

#ifndef IK_TRUE_STR
# define IK_TRUE_STR "TRUE"
#endif
#ifndef IK_FALSE_STR
# define IK_FALSE_STR "FALSE"
#endif
#ifndef IK_ON_STR
# define IK_ON_STR "ON"
#endif
#ifndef IK_OFF_STR
# define IK_OFF_STR "OFF"
#endif
#ifndef IK_BOOL_STR
# define IK_BOOL_STR(b) (b ? IK_TRUE_STR : IK_FALSE_STR)
#endif
#ifndef IK_BOOL_STR2
# define IK_BOOL_STR2(b) (b ? IK_ON_STR : IK_OFF_STR)
#endif
