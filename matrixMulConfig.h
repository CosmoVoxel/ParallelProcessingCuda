#pragma once

// CONFIGURATION FOR MATRIX MULTIPLICATION
#ifndef ELEMENTS_PER_THREAD_X
#define ELEMENTS_PER_THREAD_X 1
#endif

#ifndef ELEMENTS_PER_THREAD_Y
#define ELEMENTS_PER_THREAD_Y 1
#endif

#ifndef VERIFY
#define VERIFY false
#endif

constexpr int elements_per_thread_x = ELEMENTS_PER_THREAD_X;
constexpr int elements_per_thread_y = ELEMENTS_PER_THREAD_Y;
constexpr bool verify = VERIFY;