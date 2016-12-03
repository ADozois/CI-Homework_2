
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

FILE *file_log;

void parseFile(char *file, Data *training, Data *test);

void parseTrainingLine(char *line, Data *data);

void parseTestLine(char *line, Data *data);

void initialiseNeuron(Neuron *neuron);

int computeActivation(Neuron *neuron, double input1, double input2);

double error(Neuron *neuron, int value);

double gradient(Neuron *neuron, int value);

double gradientSigmoid(double input);

void weightUpdate(Neuron* neuron, Data* data);

double sigmoidFunc(double input);

void trainNeuron(Neuron* neuron, Data* training);

int classify(double value);

double linearFunc(double value);

void createLogFile(void);

int main(void) {
  char *path = "/home/gemini/TUM/CI/CI-Homework_2/Problem_1/testInput10A.txt";
  Data training[1000], test[100];
  Neuron my_neuron;
  srand((unsigned) time(NULL)); //See initialisation

  initialiseNeuron(&my_neuron);
  my_neuron.Func = &linearFunc;
  parseFile(path, training, test);

  trainNeuron(&my_neuron, training);

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

int computeActivation(Neuron *neuron, double input1, double input2) {
  double sum = 0.0;

  sum = neuron->Weight1*input1 + neuron->Weight2*input2 + neuron->Weight3;
  neuron->Outpout = classify(sum);
  return neuron->Outpout;
}

void initialiseNeuron(Neuron *neuron) {
  neuron->Outpout = 0;
  neuron->Weight1 = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  neuron->Weight2 = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  neuron->Weight3 = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  neuron->Func = NULL;
}

double error(Neuron *neuron, int value) {
  double error = 0.0, diff;
  diff = (double) (neuron->Outpout - value);
  error = pow(diff, 2.0);
  return error;
}

double gradient(Neuron *neuron, int value){
  return 0.0;
}

double gradientSigmoid(double input){
  return sigmoidFunc(input)*(1-sigmoidFunc(input));
}

void weightUpdate(Neuron* neuron, Data* data){
  neuron->Weight1 = neuron->Weight1 + LEARNING_WEIGHT*(gradient(neuron,data->class))*data->Input1;
  neuron->Weight2 = neuron->Weight2 + LEARNING_WEIGHT*(gradient(neuron,data->class))*data->Input2;
  neuron->Weight3 = neuron->Weight3 + LEARNING_WEIGHT*(gradient(neuron,data->class));
}

double sigmoidFunc(double input){
  return 1.0/(1 + exp((-1)*input));
}

void trainNeuron(Neuron* neuron, Data* training){
  int result = 0;
  double err = 0.0;

  for (int i = 0; i < training[0].size; ++i) {
    result = computeActivation(neuron,training[i].Input1,training[i].Input2);
    err = error(neuron,result);
    printf("%f\n", err);
    weightUpdate(neuron,training);
  }
}
int classify(double value){
  return (value > THRESHOLD)? 1 : -1;
}

double linearFunc(double value) {
  return value;
}

void createLogFile(void){
  char* path = "/home/gemini/TUM/CI/CI-Homework_2/Problem_1/log.txt";

  file_log = fopen(path,"w");
  if(!file_log){
    printf("Cannot create log file");
  }
}