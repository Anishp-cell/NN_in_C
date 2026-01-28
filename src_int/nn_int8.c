#include "nn_int8.h"

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
) {
    for (int o = 0; o < out_dim; o++) {
        int32_t acc = bias[o];

        for (int j = 0; j < in_dim; j++) {
            acc += (int32_t)((uint8_t)input[j]) * (int32_t)weights[o * in_dim + j];
        }

        float scaled = acc * input_scale * weight_scale / output_scale;

        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -128.0f) scaled = -128.0f;

        output[o] = (int8_t)(scaled);
    }
}

void relu_int8(int8_t *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0)
            x[i] = 0;
    }
}
