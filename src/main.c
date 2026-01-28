#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#include "nn.h"
#include <string.h>


#define TRAIN_SAMPLES 10000
#define TEST_SAMPLES 2000
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

float d_weights1[HIDDEN_DIM*INPUT_DIM]; // gradients for first layer
float d_bias1[HIDDEN_DIM];
float d_input1[INPUT_DIM];


int main() {
    // float *images = malloc(N * 784 * sizeof(float));
    // int *labels = malloc(N * sizeof(int));
    // load_mnist_images("./Dataset/train-images.idx3-ubyte", images, N);
    // load_mnist_labels("./Dataset/train-labels.idx1-ubyte", labels, N);
    float *train_images = malloc(TRAIN_SAMPLES * INPUT_DIM * sizeof(float));
    int   *train_labels = malloc(TRAIN_SAMPLES * sizeof(int));

    float *test_images = malloc(TEST_SAMPLES * INPUT_DIM * sizeof(float));
    int   *test_labels = malloc(TEST_SAMPLES * sizeof(int));
    load_mnist_images("./Dataset/train-images.idx3-ubyte", train_images, TRAIN_SAMPLES);
    load_mnist_labels("./Dataset/train-labels.idx1-ubyte", train_labels, TRAIN_SAMPLES);

    load_mnist_images("./Dataset/t10k-images.idx3-ubyte", test_images, TEST_SAMPLES);
    load_mnist_labels("./Dataset/t10k-labels.idx1-ubyte", test_labels, TEST_SAMPLES);
#define EPOCHS 15
    float lr = 0.01f;

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

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    float epoch_loss = 0.0f;
    // Shuffle training data
    for (int i = TRAIN_SAMPLES - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int tmp = train_labels[i];
    train_labels[i] = train_labels[j];
    train_labels[j] = tmp;

    for (int k = 0; k < INPUT_DIM; k++) {
        float t = train_images[i*INPUT_DIM + k];
        train_images[i*INPUT_DIM + k] = train_images[j*INPUT_DIM + k];
        train_images[j*INPUT_DIM + k] = t;
    }   
}
    for (int n = 0; n < TRAIN_SAMPLES; n++) {
        float *x = &train_images[n * INPUT_DIM];
        int y = train_labels[n];

    //forward pass for first layer
    dense_layer_forward(
        x,          // input vector (first image)
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
    // float sum = 0.0f;
    // for (int i = 0; i < OUTPUT_DIM; i++) sum += logits[i];
    //printf("Softmax sum = %f\n", sum);
 

    float loss = cross_entropy_loss(logits, y);
    epoch_loss += loss;

    // printf("Loss: %f\n", loss);
    if (n == 0) {
    printf("Epoch %d | First-sample loss: %f\n", epoch, loss);  
    }

    // dL/dz = softmax - one_hot(label)
    for (int i = 0; i < OUTPUT_DIM; i++) {
        d_logits[i] = logits[i];
    }
    d_logits[y] -= 1.0f;

    memset(d_weights3,0,sizeof(d_weights3));
    memset(d_bias3, 0, sizeof(d_bias3));
    //backward pass for output layer
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

    relu_backward(d_activations2, activations2, HIDDEN_DIM2);

    memset(d_weights2, 0, sizeof(d_weights2));
    memset(d_bias2, 0, sizeof(d_bias2));
    //backward pass for second layer
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

    relu_backward(d_activations1, activations1, HIDDEN_DIM);

    memset(d_weights1,0,sizeof(d_weights1));
    memset(d_bias1, 0, sizeof(d_bias1));
    //backward pass for first layer
    dense_layer_backward(
        x,          // input vector (first image)
        d_activations1,     // gradient from output
        d_input1,           // output gradient
        d_weights1,         // weight gradients
        d_bias1,            // bias gradients
        weights1,           // weights
        INPUT_DIM,          // input dimension
        HIDDEN_DIM          // output dimension
    );

    //update weights for layer 1
    for(int i=0;i<HIDDEN_DIM*INPUT_DIM;i++){
        weights1[i]-=lr * d_weights1[i];
    }
    //update bias for layer 1
    for(int i=0;i<HIDDEN_DIM;i++){
        bias1[i]-=lr*d_bias1[i];
    }

    //update weights for layer 2
    for(int i=0;i<HIDDEN_DIM2*HIDDEN_DIM;i++){
        weights2[i]-=lr*d_weights2[i];
    }
    //update bias for lauer 2
    for(int i=0;i<HIDDEN_DIM2;i++){
        bias2[i]-=lr*d_bias2[i];
    }

    // Update weights for final layer
    for (int i = 0; i < OUTPUT_DIM * HIDDEN_DIM2; i++) {
        weights3[i] -= lr * d_weights3[i];
    }

    // Update bias for final layer
    for (int i = 0; i < OUTPUT_DIM; i++) {
        bias3[i] -= lr * d_bias3[i];
    }
}

    // printf("first layer activation:\n");
    // for(int i=0;i<10;i++){
    //     printf("%f\n", activations1[i]);
    // }
    // printf("second layer activation: \n");
    // for(int i=0;i<10;i++){
    //     printf("%f\n", activations2[i]);
    // }
    // printf("output probabilities:\n");
    // for(int i=0; i<10; i++){
    //     printf("%f\n", logits[i]);
    // }
    // printf("class label: %d\n", train_labels[0]);
    // printf("class probabilities after softmax:\n");
    // int pred = 0;
    // float best = logits[0];
    // for (int i = 0; i < OUTPUT_DIM; i++) {
    //     printf("class %d: %f\n", i, logits[i]);
    //     if (logits[i] > best) {
    //         best = logits[i];
    //         pred = i;
    //     }
    // }

    printf("Epoch %d | Avg Loss: %f\n", epoch, epoch_loss / TRAIN_SAMPLES);
    }
    int correct = 0;

    for (int n = 0; n < TEST_SAMPLES; n++) {

        float *x = &test_images[n * INPUT_DIM];

        // forward pass (NO BACKPROP)
        dense_layer_forward(x, activations1, weights1, bias1, INPUT_DIM, HIDDEN_DIM);
        relu(activations1, HIDDEN_DIM);

        dense_layer_forward(activations1, activations2, weights2, bias2, HIDDEN_DIM, HIDDEN_DIM2);
        relu(activations2, HIDDEN_DIM2);

        dense_layer_forward(activations2, logits, weights3, bias3, HIDDEN_DIM2, OUTPUT_DIM);
        softmax(logits, OUTPUT_DIM);

        // argmax
        int pred = 0;
        for (int i = 1; i < OUTPUT_DIM; i++) {
            if (logits[i] > logits[pred])
                pred = i;
        }

        if (pred == test_labels[n])
            correct++;
    }

    float accuracy = 100.0f * correct / TEST_SAMPLES;
    printf("FP32 Test Accuracy: %.2f%%\n", accuracy);
    FILE *f = fopen("weights_dump.txt", "w");

for (int i = 0; i < HIDDEN_DIM * INPUT_DIM; i++)
    fprintf(f, "%ff,", weights1[i]);

fprintf(f, "\n\n");

for (int i = 0; i < HIDDEN_DIM; i++)
    fprintf(f, "%ff,", bias1[i]);

fprintf(f, "\n\n");

for (int i = 0; i < HIDDEN_DIM2 * HIDDEN_DIM; i++)
    fprintf(f, "%ff,", weights2[i]);

fprintf(f, "\n\n");

for (int i = 0; i < HIDDEN_DIM2; i++)
    fprintf(f, "%ff,", bias2[i]);

fprintf(f, "\n\n");

for (int i = 0; i < OUTPUT_DIM * HIDDEN_DIM2; i++)
    fprintf(f, "%ff,", weights3[i]);

fprintf(f, "\n\n");

for (int i = 0; i < OUTPUT_DIM; i++)
    fprintf(f, "%ff,", bias3[i]);

fclose(f);

    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    
       return 0;
}
