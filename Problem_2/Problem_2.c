
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define WEIGHT_MAX 0.000001
#define LEARNING_WEIGHT 0.0001
#define THRESHOLD 0

typedef struct Neuron Neuron;
typedef struct Data Data;
typedef double (*functionPtr)(double);

struct Neuron {
  double Weight1;
  double Weight2;
  double Weight3;
  functionPtr Func;
  int Outpout;
};

struct Data {
  double Input1;
  double Input2;
  int size;
  int class;
};

int main2(void) {

  return 0;
}