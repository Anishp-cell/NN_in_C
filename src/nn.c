#include "nn.h"

void dense_layer_forward(const float *input, float *output, const float *weights, const float *bias, int input_dimension, int output_dimension){
    for(int i= 0; i<output_dimension;i++){
        float sum= bias[i];
        for(int j=0;j<input_dimension;j++){
            sum+=input[j]*weights[i*input_dimension+j];
        }
        output[i]=sum;
    }
}