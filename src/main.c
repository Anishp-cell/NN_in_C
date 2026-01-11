#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#include "nn.h"

#define N 5
#define INPUT_DIM 784
#define HIDDEN_DIM 128

float weights1[HIDDEN_DIM * INPUT_DIM]; //weights for first layer
float bias1[HIDDEN_DIM];
float activations1[HIDDEN_DIM];


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
    printf("first layer activation:\n");
    for(int i=0;i<10;i++){
        printf("%f\n", activations1[i]);
    }
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
