#ifndef NN_H
#define NN_H
void dense_layer_forward(const float *input, float *output, const float *weights, const float *bias, int input_dimension, int output_dimension);
void relu(float *data, int size);
void softmax(float *data, int size);
float cross_entropy_loss(const float *probs, int label);
void dense_layer_backward(const float *input, const float *d_out, float *d_input, float *d_weights, float *d_bias, const float *weights, int input_dim, int output_dim);
void relu_backward(float *d_x, const float *x, int n);
#endif