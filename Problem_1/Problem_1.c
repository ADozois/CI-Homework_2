
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define WEIGHT_MAX 0.00001
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
  int Class;
};

FILE *file_log;

void parseFile(char *file, Data *training, Data *test);

void parseTrainingLine(char *line, Data *data);

void parseTestLine(char *line, Data *data);

void initialiseNeuron(Neuron *neuron);

int computeActivation(Neuron *neuron, double input1, double input2);

double error(Neuron *neuron, int value);

double gradient(Neuron *neuron, int value, double input);

double gradientTanh(double input);

void weightUpdate(Neuron *neuron, Data *data);

double tanhFunc(double input);

double linearFunc(double value);

void trainNeuron(Neuron *neuron, Data *training, Data* test);

int classify(double value);

void createLogFile(void);

void normalizeData(Data *training, Data *test);

double minInData(Data *data);

double maxInData(Data *data);

void testNeuron(Neuron *neuron, Data *test);

void divideTraining(Data* training, Data* test);

int main(void) {
  int size = 100, i = 0, j = 0, flag = 0;
  char buff[100];
  Data training[1000], test[1000], validation[1000];
  Neuron my_neuron;
  srand((unsigned) time(NULL)); //Seed initialisation

  initialiseNeuron(&my_neuron);
  my_neuron.Func = &tanhFunc;

  while(scanf("%s",buff) == 1) {
    if (strcmp(buff, "0,0,0") == 0) {
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
  training[0].size = i;
  test[0].size = j;

  normalizeData(training, validation);

  //divideTraining(training,test);

  trainNeuron(&my_neuron, training, test);

  testNeuron(&my_neuron, validation);

  return 0;
}

void parseFile(char *path, Data *training, Data *test) {
  FILE *file = NULL;
  int size = 100, i = 0, j = 0, flag = 0;
  char buff[100];

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
  data->Input2 = strtod(token[1], NULL);
  data->Class = atoi(token[2]);
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
  data->Input2 = strtod(token[1], NULL);
  data->Class = 0;
}

int computeActivation(Neuron *neuron, double input1, double input2) {
  double sum = 0.0;

  sum = neuron->Weight1 * input1 + neuron->Weight2 * input2 + neuron->Weight3;
  neuron->Outpout = classify(neuron->Func(sum));
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

double gradient(Neuron *neuron, int value, double input) {
  return (-2) * (neuron->Outpout - value) * input;
}

double gradientTanh(double input) {
  return (1 - pow(tanhFunc(input),2));
}

void weightUpdate(Neuron *neuron, Data *data) {

  neuron->Weight1 =neuron->Weight1 + LEARNING_WEIGHT * (-1) * (gradient(neuron, data->Class,data->Input1));
  neuron->Weight2 = neuron->Weight2 + LEARNING_WEIGHT * (-1) * (gradient(neuron, data->Class,data->Input2));
  neuron->Weight3 = neuron->Weight3 + LEARNING_WEIGHT * (-1) * (gradient(neuron, data->Class,1.0));
}

double tanhFunc(double input) {
  return (exp(input) - exp((-1)*input)) / (exp(input) + exp((-1) * input));
}

void trainNeuron(Neuron *neuron, Data *training, Data* test) {
  double err = 0.0, glob_err = 0.0;
  int i;

  for (i = 0; i < training[0].size; ++i) {
    computeActivation(neuron, training[i].Input1, training[i].Input2);
    err = error(neuron, training[i].Class);
    if(err != 0.0){
      glob_err += err;
    //printf("%f\n", sqrt(glob_err/(i+1)));
	  weightUpdate(neuron, &(training[i]));
	}
  }
}
int classify(double value) {
  return (value > THRESHOLD) ? 1 : -1;
}

double linearFunc(double value) {
  return value;
}

void createLogFile(void) {
  char *path = "/home/gemini/TUM/CI/CI-Homework_2/Problem_1/log.txt";

  //file_log = fopen(path, "w");
  if (!file_log) {
    printf("Cannot create log file");
  }
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

  for (j = 0; j < training[0].size; ++j) {
    training[j].Input1 = 2 * (training[j].Input1 - min) / (max - min) - 1;
    training[j].Input2 = 2 * (training[j].Input2 - min) / (max - min) - 1;
    if (j < test[0].size) {
      test[j].Input1 = 2 * (test[j].Input1 - min) / (max - min) - 1;
      test[j].Input2 = 2 * (test[j].Input2 - min) / (max - min) - 1;
    }
  }
}

double minInData(Data *data) {
  double min = data[0].Input1;
  int i;

  for (i = 0; i < data[0].size; ++i) {
    if (data[i].Input1 < min)
      min = data[i].Input1;
    if (data[i].Input2 < min)
      min = data[i].Input2;
  }

  return min;
}

double maxInData(Data *data) {
  double max = data[0].Input1;
  int i;

  for (i = 0; i < data[0].size; ++i) {
    if (data[i].Input1 > max)
      max = data[i].Input1;
    if (data[i].Input2 > max)
      max = data[i].Input2;
  }

  return max;
}

void testNeuron(Neuron *neuron, Data *test){
  int result;
  int i;

  for (i = 0; i < test[0].size; ++i) {
    result = computeActivation(neuron, test[i].Input1, test[i].Input2);
    if (result == 1)
      printf("+%d\n", result);
    else
      printf("%d\n", result);
  }

}

void divideTraining(Data* training, Data* test){
	int split = (int) ceil(training[0].size*0.7);
	int i, j = 0;

	for (i = split; i < training[0].size; i++)
	{
		test[j].Class = training[i].Class;
		test[j].Input1 = training[i].Input1;
		test[j].Input2 = training[i].Input2;
		j++;
	}
	training[0].size = split;
	test[0].size = j;
}
