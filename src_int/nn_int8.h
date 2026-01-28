#ifndef NN_INT8_H
#define NN_INT8_H

#include <stdint.h>

void dense_int8_forward(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int8_t *output,
    int in_dim,
    int out_dim,
    float input_scale,
    float weight_scale,
    float output_scale
);

void relu_int8(int8_t *x, int size);

#endif
