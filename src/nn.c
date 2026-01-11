#include "nn.h"
#include <math.h>

void softmax(float *data, int size){
    float max= data[0];
    for(int i=1;i<size;i++)
        if(data[i]>max) max=data[i];
    float sum=0.0f;
        for(int i=0;i<size;i++){
            sum+=expf(data[i]-max);
            sum+=data[i];
        }
        for(int i=0;i<size;i++){
            data[i] /=sum;
        }

    
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