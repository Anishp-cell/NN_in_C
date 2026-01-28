#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

//big endian- most significant byte first 
// little endian- least significant byte first\
// here we convet the bigendian to little endian
uint32_t readuint32(FILE *f){
    unsigned char b[4];
    fread(b,1,4,f);
    return (b[0]<<24 | (b[1]<<16) | (b[2]<<8) | b[3]); //since MNIST is big-endian we need to convert to little-endian
}
void load_mnist_images(const char *path, float *images, int num_images){
    FILE *f= fopen(path, "rb"); //rb means open in binary mode
    if(!f){
        perror("failed to open file");
        exit(1);
    }
    readuint32(f);
    int total= readuint32(f);
    int rows= readuint32(f);
    int col= readuint32(f);
    if(num_images> total) num_images=total;
    // now we loop over all images and pixeel of each image
    for(int i=0; i<num_images;i++){
        for(int j=0; j<rows*col;j++){
            unsigned char pixel;
            fread(&pixel, 1,1,f);
            images[i*784+j]= pixel/255.0f; // converts uint to float and normalizes pixel between [0.0, 1.0]
        }
    }
    fclose(f);
}
void load_mnist_labels(const char *path, int *labels, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("Labels"); exit(1); }

    readuint32(f); // magic
    readuint32(f); // total

    for (int i = 0; i < n; i++) {
        unsigned char lb;
        fread(&lb, 1, 1, f);
        labels[i] = lb;
    }
    fclose(f);
}