
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define WEIGHT_MAX 0.0001
#define LEARNING_RATE 0.00001
#define THRESHOLD 0
#define NETWORK_SIZE 4

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
  double delta;
};

struct Data {
  double Input1;
  double Input2;
  double max;
  double min;
  int size;
  double Class;
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

double minInData(Data *data, int index);

double maxInData(Data *data, int index);

void feedForward(Network *network, double input1, double input2);

void computeActivation(Neuron *neuron, double input);

void computeLayer(Layer *layer, int index);

double linearFunc(double input);

void backPropagation(Network *network, double *output, int index);

double tanhDerivate(double input);

void updateWeights(Network *network, double *delta, int index);

void trainNetwork(Network *network, Data *train, Data *test);

void printAllW(Network* network);

void backPropagation_(Network *network, double output, int index);

void updateWeights_(Network *network);


int main(void) {
  Network network;
  int i;
  int layers[NETWORK_SIZE] = {2, 3, 3, 1};
  Data training[1000], test[1000], validation[1000];

  srand((unsigned) time(NULL)); //Seed initialisation

  network.Layers = NULL;
  network.size = NETWORK_SIZE;
  network.Layers_Info = layers;

  createNetwork(&network, network.size, layers);

  parseFile("/home/gemini/TUM/CI/CI-Homework_2/Problem_2/testInput11A.txt", training, validation);

  //normalizeData(training, validation);

  /*feedForward(&network, training[0].Input1, training[0].Input2);
  printf("%f\n",network.Layers[3].Neurons->Output);
  backPropagation(&network,&training[0].Class,NETWORK_SIZE-1);
  feedForward(&network, training[0].Input1, training[0].Input2);*/

  //printf("%f",network.Layers[3].Neurons->Output);

  /*for (int j = 0; j < training->size; ++j) {
    printf("%f,%f,%1.0f\n", training[j].Input1,training[j].Input2,training[j].Class);
  }*/

  for (i = 0; i < 10000000; ++i) {
    printf("Epoch :%d\n", i + 1);
    trainNetwork(&network, training, validation);
  }

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
  data->Class = atoi(token[2]);
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
  data->Class = 0;
}

void initialiseNeuron(Neuron *neuron, int nbrWeights, functionPtr func) {
  int i = 0;
  neuron->Output = 0;
  neuron->delta = 0;
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
                tanh, i);
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
  double min1, max1, min2, max2, tmp;
  int j;

  min1 = minInData(training,1);
  max1 = maxInData(training,1);
  min2 = minInData(training,2);
  max2 = maxInData(training,2);

  for (j = 0; j < training[0].size; ++j) {
    training[j].Input1 = 2 * ((training[j].Input1 - min1) / (max1 - min1)) - 1;
    training[j].Input2 = 2 * ((training[j].Input2 - min2) / (max2 - min2)) - 1;
    if (j < test[0].size) {
      test[j].Input1 = 2 * ((test[j].Input1 - min1) / (max1 - min1)) - 1;
      test[j].Input2 = 2 * ((test[j].Input2 - min2) / (max2 - min2)) - 1;
    }
  }
}

double minInData(Data *data, int index) {
  double min1 = data[0].Input1, min2 = data[0].Input2;
  int i;

  for (i = 0; i < data[0].size; ++i) {
    if (data[i].Input1 < min1)
      min1 = data[i].Input1;
    if (data[i].Input2 < min2)
      min2 = data[i].Input2;
  }

  if (index == 1)
    return min1;
  else
    return min2;
}

double maxInData(Data *data, int index) {
  double max1 = data[0].Input1, max2 = data[0].Input2;
  int i;

  for (i = 0; i < data[0].size; ++i) {
    if (data[i].Input1 > max1)
      max1 = data[i].Input1;
    if (data[i].Input2 > max1)
      max2 = data[i].Input2;
  }

  if (index == 1)
    return max1;
  else
    return max2;
}

void feedForward(Network *network, double input1, double input2) {
  int i;
  /// Input layer
  computeActivation(&(network->Layers[0].Neurons[0]), input1);
  computeActivation(&(network->Layers[0].Neurons[1]), input2);
  computeActivation(&(network->Layers[0].Neurons[2]), 1.0);
  /// Hidden layer & output
  for (i = 1; i < network->size; ++i) {
    computeLayer(&(network->Layers[i]), i);
  }
  /*if (network->Layers[network->size-1].Neurons[0].Output > 0)
    network->Layers[network->size-1].Neurons[0].Output = 1;
  else
    network->Layers[network->size-1].Neurons[0].Output = -1;*/
}

void computeActivation(Neuron *neuron, double input) {
  neuron->Output = neuron->Func(input);
}

void computeLayer(Layer *layer, int index) {
  int i, j, size;
  double sum = 0.0, weight, output;
  size = layer->size;
  if (index != NETWORK_SIZE-1){
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
    delta[0] = (2) * (*output - network->Layers[index].Neurons[0].Output);
  } else{
    if(index == NETWORK_SIZE-2){
      size = network->Layers[index+1].size;
    } else{
      size = network->Layers[index+1].size-1;
    }
    for (i = 0; i < network->Layers[index].size; ++i) { /// Neurons current layer
      for (j = 0; j < size; ++j) { /// Neurons following layer
        delta_prev = output[j];
        w_prev = network->Layers[index+1].Neurons[j].Weights[i];
        deriv = (1 - pow(network->Layers[index].Neurons[i].Output, 2));
        sum += delta_prev * w_prev * deriv;
      }
      delta[i] = sum;
      sum = 0.0;
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
      weight_delta = LEARNING_RATE * delta[i] * output;
      network->Layers[index].Neurons[i].Weights[j] += weight_delta;
    }
  }
}

void trainNetwork(Network *network, Data *train, Data *test) {
  int i;
  double err = 0.0, sum_err = 0.0, output;

  for (i = 0; i < 2; ++i) {
    feedForward(network, train[i].Input1, train[i].Input2);
    output = network->Layers[NETWORK_SIZE - 1].Neurons->Output;
    printf("Output: %1.25f\n", output);
    err = pow((train[i].Class - network->Layers[NETWORK_SIZE - 1].Neurons->Output), 2);
    sum_err += err;
    printf("Error: %f\n", sum_err / (i+1));
    backPropagation_(network, train[i].Class, network->size - 1);
    updateWeights_(network);
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

void backPropagation_(Network *network, double output, int index){
  int i, j, k, size_actual, size_next;
  double sum = 0.0, weight_prev = 0.0, delta_prev = 0.0, output_neuron = 0.0, deriv = 0.0;

  /// Last layer
  if (index == NETWORK_SIZE-1){
   network->Layers[index].Neurons[0].delta =  (2) * (output - network->Layers[index].Neurons[0].Output);
  }
  index -= 1;
  /// Hidden layers
  for (k = index; k > 0; --k) {
    for (i = 0; i < network->Layers_Info[k]; ++i) {
      for (j = 0; j < network->Layers_Info[k + 1]; ++j) {
        weight_prev = network->Layers[k+1].Neurons[j].Weights[i];
        delta_prev = network->Layers[k+1].Neurons[j].delta;
        sum += weight_prev * delta_prev;
      }
      output_neuron = network->Layers[k].Neurons[i].Output;
      deriv = (1 - pow(output_neuron,2.0));
      network->Layers[k].Neurons[i].delta = sum * deriv;
      sum = 0.0;
    }
  }
}

void updateWeights_(Network *network){
  int i, j, k, index;
  double delta, out;

  index = NETWORK_SIZE-1;

  for (k = index; k > 0; --k) {
    for (i = 0; i < network->Layers_Info[k]; ++i) {
      for (j = 0; j < network->Layers[k - 1].size; ++j) {
        if (j != network->Layers[k - 1].size-1) {
          out = network->Layers[k].Neurons[j].Output;
        } else{
          out = 1;
        }
        delta = network->Layers[k].Neurons[i].delta;
        network->Layers[k].Neurons[i].Weights[j] += delta * out * LEARNING_RATE;
      }
    }
  }
}
