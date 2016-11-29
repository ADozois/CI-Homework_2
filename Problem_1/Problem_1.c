
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Neuron Neuron;
typedef struct Data Data;
typedef void (*functionPtr)(Neuron);

struct Neuron {
  double Input1;
  double Input2;
  functionPtr Func;
  int Outpout;
};

struct Data {
  double Input1;
  double Input2;
  int class;
};

void parseFile(char *file, Data *training, Data *test);

void parseTrainingLine(char *line, Data *data);

int main(void) {
  char *path = "/home/gemini/TUM/CI/CI-Homework_2/Problem_1/testInput10A.txt";
  Data training[1000], test[100];
  parseFile(path, training, test);

  return 0;
}

void parseFile(char *path, Data *training, Data *test) {
  FILE *file;
  int size = 100, i = 0, flag = 0;
  char buff[size];

  file = fopen(path, "r");

  if (file) {
    while (fgets(buff, size, (FILE *) file) != NULL) {
      if (strcmp(buff,"0,0,0\n") == 0) {
        flag = 1;
        printf("found");
      }
      if (flag == 0) {
        parseTrainingLine(buff, &(training[i]));
        i++;
      }
    }
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