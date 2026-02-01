#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_FILENAME 256


uint8_t* read_pgm(const char* filename, int *width, int *height) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen"); exit(1); }

    char magic[3]; fscanf(f, "%2s", magic);
    if (strcmp(magic, "P5") != 0) { fprintf(stderr, "Not a P5 PGM\n"); exit(1); }

    int w, h, maxval;
    fscanf(f, "%d %d %d\n", &w, &h, &maxval);

    *width = w; *height = h;
    uint8_t *data = malloc(w*h);
    fread(data, 1, w*h, f);
    fclose(f);
    return data;
}

void write_pgm(const char* filename, uint8_t* data, int width, int height) {
    FILE *f = fopen(filename,"wb");
    if (!f) { perror("fopen"); exit(1); }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width*height, f);
    fclose(f);
}


void convolve(uint8_t* input, uint8_t* output, int width, int height, float* kernel, int ksize) {
    int khalf = ksize / 2;
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            float sum = 0.0f;
            for (int ky=0; ky<ksize; ky++) {
                for (int kx=0; kx<ksize; kx++) {
                    int ix = x + kx - khalf;
                    int iy = y + ky - khalf;
                    if (ix >=0 && ix<width && iy >=0 && iy<height) {
                        sum += input[iy*width + ix] * kernel[ky*ksize + kx];
                    }
                }
            }
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[y*width + x] = (uint8_t)(sum + 0.5f);
        }
    }
}


int main() {

    const char* images[] = {"input_256.pgm","input_512.pgm","input_1024.pgm"};
    const char* folders[] = {"out_256","out_512","out_1024"};
    int n_images = 3;


    for(int i=0;i<n_images;i++){
        mkdir(folders[i], 0777);  // ignore if exists
    }


    float identity[9] = {0,0,0,0,1,0,0,0,0};
    float edge1[9] = {0,-1,0,-1,4,-1,0,-1,0};
    float edge2[9] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
    float sharpen3[9] = {0,-1,0,-1,5,-1,0,-1,0};
    float box_blur[9] = {1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9};
    float gaussian3[9] = {1.0f/16,2.0f/16,1.0f/16,2.0f/16,4.0f/16,2.0f/16,1.0f/16,2.0f/16,1.0f/16};


    float gaussian5[25] = {
        1,4,6,4,1,
        4,16,24,16,4,
        6,24,36,24,6,
        4,16,24,16,4,
        1,4,6,4,1
    };
    for(int i=0;i<25;i++) gaussian5[i]/=256.0f;

    float unsharp5[25] = {
        1,4,6,4,1,
        4,16,24,16,4,
        6,24,-476,24,6,
        4,16,24,16,4,
        1,4,6,4,1
    };
    for(int i=0;i<25;i++) unsharp5[i]/=-256.0f;


    float gaussian7[49] = {
        0,0,1,2,1,0,0,
        0,3,13,22,13,3,0,
        1,13,59,97,59,13,1,
        2,22,97,159,97,22,2,
        1,13,59,97,59,13,1,
        0,3,13,22,13,3,0,
        0,0,1,2,1,0,0
    };
    for(int i=0;i<49;i++) gaussian7[i]/=1003.0f;

  
    struct {char* name; float* kernel; int ksize;} filters[] = {
        {"identity", identity, 3},
        {"edge1", edge1, 3},
        {"edge2", edge2, 3},
        {"sharpen3", sharpen3, 3},
        {"box_blur", box_blur, 3},
        {"gaussian3", gaussian3, 3},
        {"gaussian5", gaussian5, 5},
        {"unsharp5", unsharp5, 5},
        {"gaussian7", gaussian7, 7}
    };
    int n_filters = sizeof(filters)/sizeof(filters[0]);

    char outname[MAX_FILENAME];


    for(int img_idx=0; img_idx<n_images; img_idx++){
        int width, height;
        uint8_t *input = read_pgm(images[img_idx], &width, &height);
        uint8_t *output = malloc(width*height);

        printf("\nProcessing %s (%dx%d)\n", images[img_idx], width, height);

        for(int f=0; f<n_filters; f++){
            clock_t start = clock();
            convolve(input, output, width, height, filters[f].kernel, filters[f].ksize);
            clock_t end = clock();
            double sec = (double)(end-start)/CLOCKS_PER_SEC;

            snprintf(outname, MAX_FILENAME, "%s/out_%s.pgm", folders[img_idx], filters[f].name);
            write_pgm(outname, output, width, height);
            printf("Saved %s | Time: %.3f sec\n", outname, sec);
        }

        free(input);
        free(output);
    }

    return 0;
}
