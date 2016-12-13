
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define WEIGHT_MAX 0.000001
#define LEARNING_WEIGHT 0.0001
#define THRESHOLD 0
#define NETWORK_SIZE 5

typedef struct Neuron Neuron;
typedef struct Data Data;
typedef double (*functionPtr)(double);
typedef struct Layer Layer;
typedef struct Network Network;

struct Network {
  Layer *Layers;
  int size;
  int *Layers_Info;
};

struct Layer {
  Neuron *Neurons;
  int size;
  Layer *Next;
  Layer *Previous;
};

struct Neuron {
  double *Weights;
  functionPtr Func;
  double Output;
};

struct Data {
  double Input;
  int size;
  double Output;
};

void parseFile(char *file, Data *training, Data *test);

void parseTrainingLine(char *line, Data *data);

void parseTestLine(char *line, Data *data);

void initialiseNeuron(Neuron *neuron, int nbrWeights, functionPtr func);

void createNetwork(Network *network, int nbrLayers, int *nbrNodes);

void createInputLayer(Network *network);

void createOutputLayer(Network *network);

void createHiddenLayer(Network *network);

void createLayer(Layer *actual, Layer *next, Layer *previous, int nbrNodes, functionPtr func);

double tanhFunc(double input);

void normalizeData(Data *training, Data *test);

void feedForward(Network *network, double input);

void computeActivation(Neuron *neuron, double input);

void computeLayer(Layer *layer);

double linearFunc(double input);

void backPropagation(Network *network, double *output, int index);

double tanhDerivate(double input);

void updateWeights(Network *network, double *delta, int index);

void trainNetwork(Network* network, Data* train, Data* test);

int main(void) {
  Network network;
  int layers[NETWORK_SIZE] = {1, 4, 3, 2, 1};
  double y = 0.02663;

  network.Layers = NULL;
  network.size = NETWORK_SIZE;
  network.Layers_Info = layers;

  createNetwork(&network, network.size, layers);

  feedForward(&network, 1.0);
  backPropagation(&network,&y,network.size-1);

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

  data->Input = strtod(token[0], NULL);
  data->Output = strtod(token[1], NULL);
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

  data->Input = strtod(token[0], NULL);
  data->Output = strtod(token[1], NULL);
}

void initialiseNeuron(Neuron *neuron, int nbrWeights, functionPtr func) {
  int i = 0;
  neuron->Output = 0;
  neuron->Weights = (double *) malloc(sizeof(double) * nbrWeights);
  for (i = 0; i < nbrWeights; ++i) {
    neuron->Weights[i] = ((double) rand() / (double) (RAND_MAX)) * WEIGHT_MAX;
  }
  neuron->Func = func;
}

void createNetwork(Network *network, int nbrLayers, int *nbrNodes) {
  network->Layers = (Layer *) malloc(sizeof(Layer) * nbrLayers);
  createInputLayer(network);
  createHiddenLayer(network);
  createOutputLayer(network);
}

void createInputLayer(Network *network) {
  createLayer(&(network->Layers[0]), &(network->Layers[1]), NULL, network->Layers_Info[0] + 1, linearFunc);
  network->Layers[0].Neurons[1].Weights[0] = 1.0;
}

void createOutputLayer(Network *network) {
  createLayer(&(network->Layers[network->size - 1]),
              NULL,
              &(network->Layers[network->size - 2]),
              network->Layers_Info[network->size - 1],
              linearFunc);
}

void createHiddenLayer(Network *network) {
  int i;
  for (i = 1; i < network->size - 1; ++i) {
    createLayer(&(network->Layers[i]),
                &(network->Layers[i + 1]),
                &(network->Layers[i - 1]),
                network->Layers_Info[i],
                tanh);
  }
}

void createLayer(Layer *actual, Layer *next, Layer *previous, int nbrNodes, functionPtr func) {
  int i;
  actual->Neurons = (Neuron *) malloc(sizeof(Neuron) * nbrNodes);
  actual->size = nbrNodes;
  actual->Next = next;
  actual->Previous = previous;
  for (i = 0; i < nbrNodes; ++i) {
    if (previous == NULL) {
      initialiseNeuron(&(actual->Neurons[i]), 1, func);
    } else {
      initialiseNeuron(&(actual->Neurons[i]), actual->Previous->size, func);
    }
  }
}

void feedForward(Network *network, double input) {
  int i;
  computeActivation(&(network->Layers[0].Neurons[0]), 1.0);
  computeActivation(&(network->Layers[0].Neurons[1]), input);
  for (i = 1; i < network->size; ++i) {
    computeLayer(&(network->Layers[i]));
  }
}

void computeActivation(Neuron *neuron, double input) {
  neuron->Output = neuron->Func(input);
}

void computeLayer(Layer *layer) {
  int i, j;
  double sum = 0.0;
  for (i = 0; i < layer->size; ++i) {
    for (j = 0; j < layer->Previous->size; ++j) {
      sum += layer->Neurons[i].Weights[j] * layer->Previous->Neurons[j].Output;
    }
    computeActivation(&(layer->Neurons[i]), sum);
  }
}

double linearFunc(double input) {
  return input;
}

void backPropagation(Network *network, double *output, int index) {
  int i, j;
  double delta[network->Layers_Info[index]];

  if (index == network->size - 1) {
    delta[0] = (-2) * (*output - network->Layers[index].Neurons[0].Output);
    backPropagation(network, delta, index - 1);
  } else {
    for (i = 0; i < network->Layers[index].size; ++i) { /// Cycle trough all node of a layer
      delta[i] = 0;
      for (j = 0; j < network->Layers[index + 1].size; ++j) { /// Cycle trough all weights of a neuron
        delta[i] += output[j] * network->Layers[index + 1].Neurons[i].Weights[j]
            * (1 - pow(network->Layers[index].Neurons[i].Output, 2));
      }
    }
    if (index > 0) {
      backPropagation(network, delta, index - 1);
    }
  }
  updateWeights(network, delta, index);
}

double tanhDerivate(double input) {
  return 1 - pow(tanh(input), 2);
}

void updateWeights(Network *network, double *delta, int index) {
  int i;

  for (i = 0; i < network->Layers[index].size; ++i) {
    network->Layers[index].Neurons[0].Weights[i] +=
        LEARNING_WEIGHT * delta[i] * network->Layers[index].Neurons[0].Output;
  }
}

void trainNetwork(Network* network, Data* train, Data* test){
  int i;

  for (i = 0; i < train[0].size; ++i) {
    feedForward(network, train[i].Output);
    backPropagation(network, &(train[i].Output),network->size-1);
  }
}