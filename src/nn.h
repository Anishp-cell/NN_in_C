#ifndef NN_H
#define NN_H
void dense_layer_forward(const float *input, float *output, const float *weights, const float *bias, int input_dimension, int output_dimension);
void relu(float *data, int size);
void softmax(float *data, int size);
#endif