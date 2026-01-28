#ifndef TRAINED_WEIGHTS_H
#define TRAINED_WEIGHTS_H

#define INPUT_DIM 784
#define HIDDEN_DIM 128
#define HIDDEN_DIM2 64
#define OUTPUT_DIM 10

float fp32_weights1[HIDDEN_DIM * INPUT_DIM] = {
    // PASTE weights1 here
};

float fp32_bias1[HIDDEN_DIM] = {
    // PASTE bias1 here
};

float fp32_weights2[HIDDEN_DIM2 * HIDDEN_DIM] = {
    // PASTE weights2 here
};

float fp32_bias2[HIDDEN_DIM2] = {
    // PASTE bias2 here
};

float fp32_weights3[OUTPUT_DIM * HIDDEN_DIM2] = {
    // PASTE weights3 here
};

float fp32_bias3[OUTPUT_DIM] = {
    // PASTE bias3 here
};

#endif
