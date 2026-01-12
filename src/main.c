#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#include "nn.h"
#include <string.h>


#define N 5
#define INPUT_DIM 784
#define HIDDEN_DIM 128
#define HIDDEN_DIM2 64
#define OUTPUT_DIM 10


float weights1[HIDDEN_DIM * INPUT_DIM]; //weights for first layer
float bias1[HIDDEN_DIM];
float activations1[HIDDEN_DIM];

float weights2[HIDDEN_DIM2 * HIDDEN_DIM]; //weights for second layer
float bias2[HIDDEN_DIM2];
float activations2[HIDDEN_DIM2];

float weights3[OUTPUT_DIM * HIDDEN_DIM2]; //weights for output layer
float bias3[OUTPUT_DIM];
float logits[OUTPUT_DIM];

float d_logits[OUTPUT_DIM]; //gradients for final layer
float d_weights3[OUTPUT_DIM *HIDDEN_DIM2];
float d_bias3[OUTPUT_DIM];
float d_activations2[HIDDEN_DIM2];

float d_weights2[HIDDEN_DIM2* HIDDEN_DIM]; // gradients for second layer
float d_bias2[HIDDEN_DIM2];
float d_activations1[HIDDEN_DIM];


int main() {
    float *images = malloc(N * 784 * sizeof(float));
    int   *labels = malloc(N * sizeof(int));
    load_mnist_images("./Dataset/train-images.idx3-ubyte", images, N);
    load_mnist_labels("./Dataset/train-labels.idx1-ubyte", labels, N);
    //initialize weights and bias with random values(first layer)
    for(int i=0; i<HIDDEN_DIM*INPUT_DIM;i++){
        weights1[i]= ((float)rand()/RAND_MAX - 0.5f) * 0.01f;
    }
    for(int i=0; i<HIDDEN_DIM;i++){
        bias1[i]=0.0f;
    }
    // initialize weights and bias with random values(second layer)
    for(int i=0;i<HIDDEN_DIM2*HIDDEN_DIM;i++){ 
        weights2[i]=((float)rand()/RAND_MAX - 0.05f) * 0.01f;
    }
    for(int i=0; i<HIDDEN_DIM2;i++){
        bias2[i]=0.0f;
    }
    // initialize weights and bias with random values(output layer)
    for(int i= 0;i<OUTPUT_DIM*HIDDEN_DIM2;i++){
        weights3[i]= ((float)rand()/RAND_MAX - 0.1f) * 0.01f;
    }
    for(int i=0; i<OUTPUT_DIM;i++){
        bias3[i]= 0.0f;
    }

    for(int step =0; step<5;step++){

    //forward pass for first layer
    dense_layer_forward(
        &images[0],          // input vector (first image)
        activations1,        // output vector
        weights1,            // weights
        bias1,               // bias
        INPUT_DIM,           // input dimension
        HIDDEN_DIM           // output dimension
    );
    relu(activations1, HIDDEN_DIM);
    //forward pass for second layer
    dense_layer_forward(
        activations1,       // input vector
        activations2,        // output vector
        weights2,            // weights
        bias2,               // bias
        HIDDEN_DIM,          // input dimension
        HIDDEN_DIM2          // output dimension
    );
    relu(activations2, HIDDEN_DIM2);
    //forward pass for output layer
    dense_layer_forward(
        activations2,       // input vector
        logits,             // output vector
        weights3,           // weights
        bias3,              // bias
        HIDDEN_DIM2,       // input dimension
        OUTPUT_DIM         // output dimension
    );
    softmax(logits, OUTPUT_DIM);
    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_DIM; i++) sum += logits[i];
    printf("Softmax sum = %f\n", sum);
 
    float loss = cross_entropy_loss(logits, labels[0]);
    printf("Loss: %f\n", loss);
    // dL/dz = softmax - one_hot(label)
    for (int i = 0; i < OUTPUT_DIM; i++) {
        d_logits[i] = logits[i];
    }
    d_logits[labels[0]] -= 1.0f;

    memset(d_weights3,0,sizeof(d_weights3));
    memset(d_bias3, 0, sizeof(d_bias3));
    dense_layer_backward(
        activations2,       // input vector
        d_logits,           // gradient from output
        d_activations2,     // output gradient
        d_weights3,         // weight gradients
        d_bias3,            // bias gradients
        weights3,           // weights
        HIDDEN_DIM2,       // input dimension
        OUTPUT_DIM         // output dimension
    );
    float lr = 0.01f;
    relu_backward(d_activations2, activations2, HIDDEN_DIM2);
    memset(d_weights2, 0, sizeof(d_weights2));
    memset(d_bias2, 0, sizeof(d_bias2));
    dense_layer_backward(
        activations1,       // input vector
        d_activations2,     // gradient from output
        d_activations1,     // output gradient
        d_weights2,         // weight gradients
        d_bias2,            // bias gradients
        weights2,           // weights
        HIDDEN_DIM,         // input dimension
        HIDDEN_DIM2         // output dimension
    );

    //update weights for layer 2
    for(int i=0;i<HIDDEN_DIM2*HIDDEN_DIM;i++){
        weights2[i]-=lr*d_weights2[i];
    }
    //update bias for lauer 2
    for(int i=0;i<HIDDEN_DIM2;i++){
        bias2[i]-=lr*d_bias2[i];
    }
    // Update weights for layer 1
    for (int i = 0; i < OUTPUT_DIM * HIDDEN_DIM2; i++) {
        weights3[i] -= lr * d_weights3[i];
    }

    // Update bias for layer 1
    for (int i = 0; i < OUTPUT_DIM; i++) {
        bias3[i] -= lr * d_bias3[i];
    }
}

    printf("first layer activation:\n");
    for(int i=0;i<10;i++){
        printf("%f\n", activations1[i]);
    }
    printf("second layer activation: \n");
    for(int i=0;i<10;i++){
        printf("%f\n", activations2[i]);
    }
    printf("output probabilities:\n");
    for(int i=0; i<10; i++){
        printf("%f\n", logits[i]);
    }
    printf("class label: %d\n", labels[0]);
    printf("class probabilities after softmax:\n");
    int pred = 0;
    float best = logits[0];
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("class %d: %f\n", i, logits[i]);
        if (logits[i] > best) {
            best = logits[i];
            pred = i;
        }
    }

    printf("predicted digit: %d\n", pred);
    printf("true label: %d\n", labels[0]);


    // for (int i = 0; i < N; i++) {
    //     printf("Label: %d\n", labels[i]);
    //     for (int r = 0; r < 28; r++) {
    //         for (int c = 0; c < 28; c++) {
    //             float v = images[i*784 + r*28 + c];
    //             printf("%c", v > 0.5 ? '#' : ' ');
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    free(images);
    free(labels);
    return 0;
}
