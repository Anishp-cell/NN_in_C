#include "nn.h"
#include <math.h>

void relu_backward(float *d_x, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] <= 0)
            d_x[i] = 0.0f;
    }
}
void dense_layer_backward(const float *input, const float *d_out, float *d_input, float *d_weights, float *d_bias, const float *weights, int input_dim, int output_dim){
    //compute bias gradients for output layer
    for(int i=0;i<output_dim;i++){
        d_bias[i]= d_out[i];
    }
    //compute weight gradients for output layer
    for(int i=0;i<output_dim;i++){
        for(int j=0; j<input_dim;j++){
            d_weights[i*input_dim+j]+= d_out[i]*input[j];
        }
    }
    //compute input gradients for output layer
    for(int i=0; i<input_dim;i++){
        float sum=0.0f;
        for(int j=0; j<output_dim;j++){
            sum+= weights[j*input_dim+i]*d_out[j];
        }
        d_input[i]=sum;
    }
}
float cross_entropy_loss(const float *probs, int label){
    return -logf(probs[label]);
}


void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

void dense_layer_forward(const float *input, float *output, const float *weights, const float *bias, int input_dimension, int output_dimension){
    for(int i= 0; i<output_dimension;i++){
        float sum= bias[i];
        for(int j=0;j<input_dimension;j++){
            sum+=input[j]*weights[i*input_dimension+j];
        }
        output[i]=sum;
    }
}
void relu(float *data, int size){
    for(int i=0;i<size;i++){
        if(data[i]<0){
            data[i]=0;
        }
    }
}