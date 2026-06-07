/* Auto-generated test data for YuNet postprocessing verification.
   Generated from stedgeai_validation.npz
   Input: NCHW BGR float32, 320x320
   Outputs: 12 tensors (cls, obj, bbox, kps ¡Á 3 strides)
*/
#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <stdint.h>
#define TEST_IMG_H 320
#define TEST_IMG_W 320
#define TEST_IMG_C 3
#define TEST_NUM_OUTPUTS 12

extern const float test_input[TEST_IMG_C * TEST_IMG_H * TEST_IMG_W];
extern const float test_output_0[16000];
extern const float test_output_1[4000];
extern const float test_output_2[1000];
extern const float test_output_3[16000];
extern const float test_output_4[4000];
extern const float test_output_5[1000];
extern const float test_output_6[64000];
extern const float test_output_7[16000];
extern const float test_output_8[4000];
extern const float test_output_9[160000];
extern const float test_output_10[40000];
extern const float test_output_11[10000];

#endif /* TEST_DATA_H */
