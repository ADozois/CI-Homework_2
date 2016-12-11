
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
typedef struct Layer Layer;
typedef struct Network Network;

struct Network{
  Layer* Layers;
  int size;
  int* Layers_Info;
};

struct Layer{
  Neuron* Neurons;
  int size;
  Layer* Next;
  Layer* Previous;
};

struct Neuron {
  double* Weights;
  functionPtr Func;
  int Output;
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

void initialiseNeuron(Neuron *neuron, int nbrWeights, functionPtr func);

void createNetwork(Network* network, int nbrLayers, int* nbrNodes);

void createInputLayer(Network* network);

void createOutputLayer(Network* network);

void createHiddenLayer(Network* network);

void createLayer(Layer* actual, Layer* next, Layer* previous, int nbrNodes);

double tanhFunc(double input);


int main(void) {
  Network network;
  int layers[3] = {2,3,1};

  network.Layers = NULL;
  network.size = 3;
  network.Layers_Info = layers;

  createNetwork(&network, network.size, layers);


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

void initialiseNeuron(Neuron *neuron, int nbrWeights, functionPtr func) {
  int i = 0;
  neuron->Output = 0;
  neuron->Weights = (double*) malloc(sizeof(double)*nbrWeights);
  for (i = 0; i < nbrWeights; ++i) {
    neuron->Weights[i] = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  }
  neuron->Func = func;
}

void createNetwork(Network* network, int nbrLayers, int* nbrNodes){
  network->Layers = (Layer*) malloc(sizeof(Layer)*nbrLayers);
  createInputLayer(network);
  createOutputLayer(network);
  createHiddenLayer(network);
}

void createInputLayer(Network* network){
  createLayer(&(network->Layers[0]),&(network->Layers[1]),NULL,network->Layers_Info[0]);
}

void createOutputLayer(Network* network){
  createLayer(&(network->Layers[network->size-1]),NULL,&(network->Layers[network->size-2]),network->Layers_Info[network->size-1]);
}

void createHiddenLayer(Network* network){
  int i;
  for(i=1; i<network->size-1; ++i){
    createLayer(&(network->Layers[i]),&(network->Layers[i+1]),&(network->Layers[i-1]),network->Layers_Info[network->size-1]);
  }
}

void createLayer(Layer* actual, Layer* next, Layer* previous, int nbrNodes){
  int i;
  actual->Neurons = (Neuron*) malloc(sizeof(Neuron)*nbrNodes);
  for (i = 0; i < nbrNodes; ++i){
    initialiseNeuron(&(actual->Neurons[i]), 1, tanh);
  }
  actual->size = nbrNodes;
  actual->Next = next;
  actual->Previous = previous;
}