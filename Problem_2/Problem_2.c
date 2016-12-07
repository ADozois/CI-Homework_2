
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

void parseFile(char *file, Data *training, Data *test);

void parseTrainingLine(char *line, Data *data);

void parseTestLine(char *line, Data *data);

void initialiseNeuron(Neuron *neuron);

int computeActivation(Neuron *neuron, double input1, double input2);

double error(Neuron *neuron, int value);


int main2(void) {

  return 0;
}

void parseFile(char *path, Data *training, Data *test) {
  FILE *file;
  int size = 100, i = 0, j = 0, flag = 0;
  char buff[size];

  file = fopen(path, "r");

  if (file) {
    while (fgets(buff, size, (FILE *) file) != NULL) {
      if (strcmp(buff, "0,0,0\n") == 0) {
        flag = 1;
      } else {
        if (flag == 0) {
          parseTrainingLine(buff, &(training[i]));
          ++i;
        } else {
          parseTestLine(buff, &(test[j]));
          ++j;
        }
      }
    }
    training[0].size = i;
    test[0].size = j;
  } else {
    printf("Can't open file");
  }
}

void parseTrainingLine(char *line, Data *data) {
  char *token[3], *ptr;
  int i = 0;

  ptr = strtok(line, ",\n");

  while (ptr != NULL) {
    token[i] = ptr;
    ptr = strtok(NULL, ",\n");
    ++i;
  }

  data->Input1 = strtod(token[0], NULL);
  data->Input2 = strtod(token[1], NULL);
  data->class = atoi(token[2]);
}

void parseTestLine(char *line, Data *data) {
  char *token[2], *ptr;
  int i = 0;

  ptr = strtok(line, ",\n");

  while (ptr != NULL) {
    token[i] = ptr;
    ptr = strtok(NULL, ",\n");
    ++i;
  }

  data->Input1 = strtod(token[0], NULL);
  data->Input2 = strtod(token[1], NULL);
  data->class = 0;
}

void initialiseNeuron(Neuron *neuron) {
  neuron->Outpout = 0;
  neuron->Weight1 = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  neuron->Weight2 = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  neuron->Weight3 = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  neuron->Func = NULL;
}

int computeActivation(Neuron *neuron, double input1, double input2) {
  double sum = 0.0;

  sum = neuron->Weight1 * input1 + neuron->Weight2 * input2 + neuron->Weight3;
  //neuron->Outpout = classify(sum);
  return neuron->Outpout;
}

double error(Neuron *neuron, int value) {
  double error = 0.0, diff;
  diff = (double) (neuron->Outpout - value);
  error = pow(diff, 2.0);
  return error;
}
