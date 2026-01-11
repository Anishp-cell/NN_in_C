#ifndef MNIST_H
#define MNIST_H
#include<stdio.h>
#include<stdint.h>

uint32_t readuint32(FILE *f);
void load_mnist_images(const char *path, float *images, int num_images);
void load_mnist_labels(const char *path, int *labels, int num_labels);
#endif 