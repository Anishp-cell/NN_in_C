#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include "mnist.h"
#include "nn_int8.h"
#include "trained_weights.h"

#define TEST_SAMPLES 2000
#define INPUT_DIM 784
#define HIDDEN_DIM 128
#define HIDDEN_DIM2 64
#define OUTPUT_DIM 10

int main() {

    float *test_images = malloc(TEST_SAMPLES * INPUT_DIM * sizeof(float));
    int   *test_labels = malloc(TEST_SAMPLES * sizeof(int));

    load_mnist_images("./Dataset/t10k-images.idx3-ubyte", test_images, TEST_SAMPLES);
    load_mnist_labels("./Dataset/t10k-labels.idx1-ubyte", test_labels, TEST_SAMPLES);

    int correct = 0;

    int8_t x_q[INPUT_DIM];
    int8_t a1[HIDDEN_DIM];
    int8_t a2[HIDDEN_DIM2];
    int8_t out[OUTPUT_DIM];


    for (int n = 0; n < TEST_SAMPLES; n++) {

        // Quantize input
        for (int i = 0; i < INPUT_DIM; i++) {
            x_q[i] = (int8_t)(test_images[n*INPUT_DIM + i] / INPUT_SCALE);
        }

        dense_int8_forward(
                        x_q,
                        W1_Q,
                        B1_Q,
                        a1,
                        INPUT_DIM,
                        HIDDEN_DIM,
                        INPUT_SCALE,
                        W1_SCALE,
                        ACT1_SCALE
                    );

        relu_int8(a1, HIDDEN_DIM);

        dense_int8_forward(
                            a1,
                            W2_Q,
                            B2_Q,
                            a2,
                            HIDDEN_DIM,
                            HIDDEN_DIM2,
                            ACT1_SCALE,
                            W2_SCALE,
                            ACT2_SCALE
                        );

        relu_int8(a2, HIDDEN_DIM2);

        dense_int8_forward(
                        a2,
                        W3_Q,
                        B3_Q,
                        out,
                        HIDDEN_DIM2,
                        OUTPUT_DIM,
                        ACT2_SCALE,
                        W3_SCALE,
                        OUTPUT_SCALE
                    );


        // Dequantize + argmax
        int pred = 0;
        float best = -1e9;

        for (int i = 0; i < OUTPUT_DIM; i++) {
            float val = out[i] * OUTPUT_SCALE;
            if (val > best) {
                best = val;
                pred = i;
            }
        }

        if (pred == test_labels[n])
            correct++;
    }

    printf("INT8 Test Accuracy: %.2f%%\n",
           100.0f * correct / TEST_SAMPLES);

    return 0;
}
