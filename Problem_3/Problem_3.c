
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
typedef double (*functionPtr)(double, Neuron*);

struct Neuron {
  double Weight1;
  double Weight2;
  functionPtr Func;
  double Output;
};

struct Data {
  double Input1;
  double max;
  double min;
  int size;
  double Output;
};

void parseFile(char *file, Data *training, Data *test);

void parseTrainingLine(char *line, Data *data);

void parseTestLine(char *line, Data *data);

void initialiseNeuron(Neuron *neuron);

double linearFunc(double value, Neuron* neuron);

void trainNeuron(Neuron *neuron, Data *training);

void computeActivation(Neuron *neuron, double input1);

double error(Neuron *neuron, double value);

void weightUpdate(Neuron *neuron, Data *data);

double gradient(Neuron *neuron, double value, double input);

void normalizeData(Data *training, Data *test);

void unnormalizeData(Data *training, Data *test);

double minInData(Data *data);

double maxInData(Data *data);

void validateNeuron(Neuron *neuron, Data *test);

void divideTraining(Data* training, Data* test);

void testNeuron(Neuron *neuron, Data *test);



int main (void){
  char* path = "/home/gemini/TUM/CI/CI-Homework_2/Problem_3/testInput12D.txt";
  char buff[100];
  int flag = 0;
  Data training[1000], validation[1000], test[100];
  Neuron neuron;
  int i = 0, j = 0;

  initialiseNeuron(&neuron);

  while(scanf("%s",buff) == 1) {
    if (strcmp(buff, "0,0") == 0) {
      flag = 1;
    } else {
      if (flag == 0) {
        parseTrainingLine(buff, &(training[i]));
        ++i;
      } else {
        parseTestLine(buff, &(validation[j]));
        ++j;
      }
    }
  }

  //parseFile(path,training,validation);

  normalizeData(training, validation);

  divideTraining(training,test);

  for (i = 0; i < 100; ++i) {
    //printf("Epoch: %d  training error: \n", (i+1));
    trainNeuron(&neuron, training);
    //printf("Epoch: %d  test error: \n", (i+1));
    testNeuron(&neuron,test);
  }

  //printf("[%f,%f]",neuron.Weight2, neuron.Weight1);

  validateNeuron(&neuron, validation);

  return 0;
}

void parseFile(char *path, Data *training, Data *test) {
  FILE *file = NULL;
  int size = 100, i = 0, j = 0, flag = 0;
  char buff[100];

  file = fopen(path, "r");

  if (file) {
    while (fgets(buff, size, (FILE *) file) != NULL) {
      if (strcmp(buff, "0,0\n") == 0) {
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
  char *token[3], *ptr = NULL;
  int i = 0;

  ptr = strtok(line, ",\n");

  while (ptr != NULL) {
    token[i] = ptr;
    ptr = strtok(NULL, ",\n");
    ++i;
  }

  data->Input1 = strtod(token[0], NULL);
  data->Output= strtod(token[1], NULL);
}

void parseTestLine(char *line, Data *data) {
  char *token[2], *ptr = NULL;
  int i = 0;

  ptr = strtok(line, ",\n");

  while (ptr != NULL) {
    token[i] = ptr;
    ptr = strtok(NULL, ",\n");
    ++i;
  }

  data->Input1 = strtod(token[0], NULL);
  data->Output= strtod(token[1], NULL);
}

void initialiseNeuron(Neuron *neuron) {
  neuron->Output = 0;
  neuron->Weight1 = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  neuron->Weight2 = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  neuron->Func = linearFunc;
}

double linearFunc(double value, Neuron* neuron){
  return neuron->Weight1 * value + neuron->Weight2;
}

void trainNeuron(Neuron *neuron, Data *training) {
  double err = 0.0, glob_err = 0.0;
  int i;

  for (i = 0; i < training[0].size; ++i) {
    computeActivation(neuron, training[i].Input1);
    err = error(neuron, training[i].Output);
    if(err != 0.0){
      glob_err += err;
      //printf("%f\n", sqrt(glob_err/(i+1)));
      weightUpdate(neuron, &(training[i]));
    }
  }
}

void computeActivation(Neuron *neuron, double input1) {
  neuron->Output = neuron->Func(input1, neuron);
}


double error(Neuron *neuron, double value) {
  double error = 0.0, diff;
  diff = (value - neuron->Output);
  error = pow(diff, 2.0);
  return error;

}

void weightUpdate(Neuron *neuron, Data *data) {

  neuron->Weight1 =neuron->Weight1 + LEARNING_WEIGHT * (-1) * (gradient(neuron, data->Output,data->Input1));
  neuron->Weight2 = neuron->Weight2 + LEARNING_WEIGHT * (-1) * (gradient(neuron, data->Output,1.0));
}

double gradient(Neuron *neuron, double value, double input) {
  return (-2) * (value - neuron->Output) * input;
}

void normalizeData(Data *training, Data *test) {
  double min, max, tmp;
  int j;

  min = minInData(training);
  max = maxInData(training);

  tmp = minInData(test);
  if (tmp < min)
    min = tmp;
  tmp = maxInData(test);
  if (tmp > max)
    max = tmp;

  training[0].max = max;
  training[0].min = min;
  test[0].max = max;
  test[0].min = min;

  for (j = 0; j < training[0].size; ++j) {
    training[j].Input1 = 2 * (training[j].Input1 - min) / (max - min) - 1;
    if (j < test[0].size) {
      test[j].Input1 = 2 * (test[j].Input1 - min) / (max - min) - 1;
    }
  }
}

double minInData(Data *data) {
  double min = data[0].Input1;
  int i;

  for (i = 0; i < data[0].size; ++i) {
    if (data[i].Input1 < min)
      min = data[i].Input1;
  }

  return min;
}

double maxInData(Data *data) {
  double max = data[0].Input1;
  int i;

  for (i = 0; i < data[0].size; ++i) {
    if (data[i].Input1 > max)
      max = data[i].Input1;
  }

  return max;
}

void validateNeuron(Neuron *neuron, Data *test){
  int i;

  for (i = 0; i < test[0].size; ++i) {
    computeActivation(neuron, test[i].Input1);
    printf("%f\n", neuron->Output);
  }

}

void divideTraining(Data* training, Data* test){
  int split = (int) ceil(training[0].size*0.7);
  int i, j = 0;

  for (i = split; i < training[0].size; i++)
  {
    test[j].Output = training[i].Output;
    test[j].Input1 = training[i].Input1;
    j++;
  }
  training[0].size = split;
  test[0].size = j;
}

void testNeuron(Neuron *neuron, Data *test){
  double err = 0.0, glob_err = 0.0;
  int i;

  for (i = 0; i < test[0].size; ++i) {
    computeActivation(neuron, test[i].Input1);
    err = error(neuron, test[i].Output);
    if(err != 0.0) {
      glob_err += err;
      //printf("%f\n", sqrt(glob_err/(i+1)));
    }
  }

}

