
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define WEIGHT_MAX 0.001
#define LEARNING_RATE 0.0001
#define THRESHOLD 0
#define NETWORK_SIZE 6

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
  double max;
  double min;
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

void createLayer(Layer *actual, Layer *next, Layer *previous, int nbrNodes, functionPtr func, int index);

double tanhFunc(double input);

void normalizeData(Data *training, Data *test);

double minInData(Data *data);

double maxInData(Data *data);

void feedForward(Network *network, double input);

void computeActivation(Neuron *neuron, double input);

void computeLayer(Layer *layer, int index);

double linearFunc(double input);

void backPropagation(Network *network, double *output, int index);

double tanhDerivate(double input);

void updateWeights(Network *network, double *delta, int index);

void trainNetwork(Network *network, Data *train, Data *test);

void printAllW(Network* network);

int main(void) {
  Network network;
  int i;
  int layers[NETWORK_SIZE] = {1, 4, 4, 3, 3, 1};
  Data training[1000], test[1000], validation[1000];

  srand((unsigned) time(NULL)); //Seed initialisation

  network.Layers = NULL;
  network.size = NETWORK_SIZE;
  network.Layers_Info = layers;

  createNetwork(&network, network.size, layers);

  parseFile("/home/gemini/TUM/CI/CI-Homework_2/Problem_4/testInput13A.txt", training, validation);

  printAllW(&network);

  normalizeData(training, validation);

  //feedForward(&network, training[0].Input);

  //printf("%f", training[0].Output - network.Layers[2].Neurons[2].Output);

  for (i = 0; i < 10000; ++i) {
    //printf("Epoch :%d\n", i + 1);
    trainNetwork(&network, training, validation);
  }

  printf("\n\n");

  printAllW(&network);

  return 0;
}

void parseFile(char *path, Data *training, Data *test) {
  FILE *file;
  int size = 100, i = 0, j = 0, flag = 0;
  char buff[size];

  file = fopen(path, "r");

  if (file) {
    while (fgets(buff, size, (FILE *) file) != NULL) {
      if (strcmp(buff, "0,0") == 0) {
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
    neuron->Weights[i] = WEIGHT_MAX * ((double)rand()/(double)RAND_MAX - 0.5);
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
  createLayer(&(network->Layers[0]), &(network->Layers[1]), NULL, network->Layers_Info[0], linearFunc, 0);
}

void createOutputLayer(Network *network) {
  createLayer(&(network->Layers[network->size - 1]),
              NULL,
              &(network->Layers[network->size - 2]),
              network->Layers_Info[network->size - 1],
              linearFunc, network->size - 1);
}

void createHiddenLayer(Network *network) {
  int i;
  for (i = 1; i < network->size - 1; ++i) {
    createLayer(&(network->Layers[i]),
                &(network->Layers[i + 1]),
                &(network->Layers[i - 1]),
                network->Layers_Info[i],
                tanhFunc, i);
  }
}

void createLayer(Layer *actual, Layer *next, Layer *previous, int nbrNodes, functionPtr func, int index) {
  if (index < NETWORK_SIZE - 1){
    actual->Neurons = (Neuron*) malloc(sizeof(Neuron)*(nbrNodes+1));
    nbrNodes += 1;
  } else{
    actual->Neurons = (Neuron*) malloc(sizeof(Neuron)*nbrNodes);
  }
  actual->size = nbrNodes;
  actual->Next = next;
  actual->Previous = previous;
  for (int i = 0; i < nbrNodes; ++i) {
    if (index == 0){
      initialiseNeuron(&(actual->Neurons[i]),1,linearFunc);
    } else if(index == NETWORK_SIZE-1){
      initialiseNeuron(&(actual->Neurons[i]),actual->Previous->size,linearFunc);
    } else{
      if (i < nbrNodes-1) {
        initialiseNeuron(&(actual->Neurons[i]), actual->Previous->size, func);
      } else{
        initialiseNeuron(&(actual->Neurons[i]), 1, linearFunc);
      }
    }
  }
}

double tanhFunc(double input) {
  return (exp(input) - exp(-input)) / (exp(input) + exp(-input));
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
    training[j].Input = 2 * (training[j].Input - min) / (max - min) - 1;
    if (j < test[0].size) {
      test[j].Input = 2 * (test[j].Input - min) / (max - min) - 1;
    }
  }
}

double minInData(Data *data) {
  double min = data[0].Input;
  int i;

  for (i = 0; i < data[0].size; ++i) {
    if (data[i].Input < min)
      min = data[i].Input;
  }

  return min;
}

double maxInData(Data *data) {
  double max = data[0].Input;
  int i;

  for (i = 0; i < data[0].size; ++i) {
    if (data[i].Input > max)
      max = data[i].Input;
  }

  return max;
}

void feedForward(Network *network, double input) {
  int i;
  /// Input layer
  computeActivation(&(network->Layers[0].Neurons[0]), input);
  computeActivation(&(network->Layers[0].Neurons[1]), 1.0);
  /// Hidden layer & output
  for (i = 1; i < network->size; ++i) {
    computeLayer(&(network->Layers[i]), i);
  }
}

void computeActivation(Neuron *neuron, double input) {
  neuron->Output = neuron->Func(input);
}

void computeLayer(Layer *layer, int index) {
  int i, j, size;
  double sum = 0.0, weight, output;
  size = layer->size;
  if (layer->Next != NULL){
    size -= 1;
    computeActivation(&(layer->Neurons[layer->size-1]),1.0);
  }
  for (i = 0; i < size; ++i) { /// For all neurons
    for (j = 0; j <layer->Previous->size ; ++j) { /// For all neurons/weights
      output = layer->Previous->Neurons[j].Output;
      weight = layer->Neurons[i].Weights[j];
      sum += weight * output;
    }
    computeActivation(&(layer->Neurons[i]),sum);
    sum = 0.0;
  }
}

double linearFunc(double input) {
  return input;
}

void backPropagation(Network *network, double *output, int index) {
  int i, j, size;
  double w_prev, delta_prev, deriv, sum = 0.0;
  double delta[network->Layers[index].size];

  if (index == NETWORK_SIZE-1){
    delta[0] = (-2) * (*output - network->Layers[index].Neurons[0].Output);
  } else{
    if(index == NETWORK_SIZE-2){
      size = network->Layers[index+1].size;
    } else{
      size = network->Layers[index+1].size-1;
    }
    for (i = 0; i < network->Layers[index].size - 1; ++i) { /// Neurons current layer
      for (j = 0; j < size; ++j) { /// Neurons following layer
        delta_prev = output[j];
        w_prev = network->Layers[index+1].Neurons[j].Weights[i];
        deriv = (1 - pow(network->Layers[index].Neurons[i].Output, 2));
        sum += delta_prev * w_prev * deriv;
      }
      delta[i] = sum;
    }
  }
  if (index > 1){
    backPropagation(network,delta,index-1);
  }
  updateWeights(network,delta,index);
}

double tanhDerivate(double input) {
  return 1 - pow(tanh(input), 2);
}

void updateWeights(Network *network, double *delta, int index) {
  int i, j, size;
  double output, weight_delta;
  if(index == NETWORK_SIZE-1){
    size = network->Layers[index].size;
  } else{
    size = network->Layers[index].size-1;
  }
  for (i = 0; i < size; ++i) {
    for (j = 0; j < network->Layers[index-1].size; ++j) {
      output = network->Layers[index-1].Neurons[j].Output;
      weight_delta = (-1)*LEARNING_RATE * delta[i] * output;
      network->Layers[index].Neurons[i].Weights[j] += weight_delta;
    }
  }
}

void trainNetwork(Network *network, Data *train, Data *test) {
  int i;
  double err = 0.0, sum_err = 0.0, output;

  for (i = 0; i < train[0].size; ++i) {
    feedForward(network, train[i].Input);
    output = network->Layers[NETWORK_SIZE - 1].Neurons->Output;
    //printf("Output: %1.25f\n", output);
    err = pow((train[i].Output - network->Layers[NETWORK_SIZE - 1].Neurons->Output), 2);
    sum_err += err;
    //printf("Error: %f\n", sum_err / (i+1));
    backPropagation(network, &(train[i].Output), network->size - 1);
  }
}


void printAllW(Network* network){
  int i, j, k;

  for (i = 1; i < network->size; ++i) { /// Index layer
    for (j = 0; j < network->Layers[i].size; ++j) {
      for (k = 0; k < network->Layers[i].Previous->size; ++k) {
        if (j < network->Layers[i].size-1 || i == NETWORK_SIZE-1) {
          printf("layer %i, neuron %i,weight %i: %f\n", i, j, k, network->Layers[i].Neurons[j].Weights[k]);
        }else{
          printf("layer %i, neuron %i,weight %i: %f\n", i, j, k, network->Layers[i].Neurons[j].Weights[0]);
        }
      }

    }
  }
}

