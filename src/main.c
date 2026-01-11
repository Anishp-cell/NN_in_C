#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#include <direct.h>   // Windows version of getcwd

#define N 5

int main() {
    float *images = malloc(N * 784 * sizeof(float));
    int   *labels = malloc(N * sizeof(int));

    load_mnist_images("./Dataset/train-images.idx3-ubyte", images, N);
    load_mnist_labels("./Dataset/train-labels.idx1-ubyte", labels, N);

    for (int i = 0; i < N; i++) {
        printf("Label: %d\n", labels[i]);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                float v = images[i*784 + r*28 + c];
                printf("%c", v > 0.5 ? '#' : ' ');
            }
            printf("\n");
        }
        printf("\n");
    }

    free(images);
    free(labels);
    return 0;
}
