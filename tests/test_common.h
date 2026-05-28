#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Test counters */
static int test_count = 0;
static int test_passed = 0;
static int test_failed = 0;

/* Assertion macros */
#define ASSERT(cond) do { \
    test_count++; \
    if (!(cond)) { \
        test_failed++; \
        printf("  ✗ FAIL at %s:%d - %s\n", __func__, __LINE__, #cond); \
    } else { \
        test_passed++; \
    } \
} while(0)

#define ASSERT_NOT_NULL(ptr) do { \
    test_count++; \
    if (!(ptr)) { \
        test_failed++; \
        printf("  ✗ FAIL at %s:%d - Expected non-NULL, got NULL\n", __func__, __LINE__); \
    } else { \
        test_passed++; \
    } \
} while(0)

#define ASSERT_NULL(ptr) do { \
    test_count++; \
    if ((ptr)) { \
        test_failed++; \
        printf("  ✗ FAIL at %s:%d - Expected NULL, got non-NULL\n", __func__, __LINE__); \
    } else { \
        test_passed++; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    test_count++; \
    if ((a) != (b)) { \
        test_failed++; \
        printf("  ✗ FAIL at %s:%d - %d != %d\n", __func__, __LINE__, (int)(a), (int)(b)); \
    } else { \
        test_passed++; \
    } \
} while(0)

#define ASSERT_EQ_FLOAT(a, b, tol) do { \
    test_count++; \
    if (fabs((a) - (b)) > (tol)) { \
        test_failed++; \
        printf("  ✗ FAIL at %s:%d - %f != %f (tol=%f)\n", __func__, __LINE__, (float)(a), (float)(b), (float)(tol)); \
    } else { \
        test_passed++; \
    } \
} while(0)

#define TEST_HEADER(name) \
    printf("\n%s\n", name)

#define PRINT_STATS() \
    printf("\n" \
    "╔════════════════════════════════════╗\n" \
    "║   Test Report Summary             ║\n" \
    "╠════════════════════════════════════╣\n" \
    "║ Total:  %d                        ║\n" \
    "║ Passed: %d ✓                       ║\n" \
    "║ Failed: %d ✗                       ║\n" \
    "╚════════════════════════════════════╝\n", \
    test_count, test_passed, test_failed)

#define TEST_RESULT() (test_failed == 0 ? 0 : 1)

#endif /* TEST_COMMON_H */
