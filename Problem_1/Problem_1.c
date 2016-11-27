
#include <stdio.h>

typedef struct Neuron Neuron;
typedef struct Data Data;
typedef void (*functionPtr)(Neuron);

struct Neuron{
    double Input1;
    double Input2;
    functionPtr Func;
    int Outpout;
};

struct Data{
    double Input1;
    double Input2;
    int class;
};

void parseFile(char* file, Data** training, Data** test);

void parseTrainingLine(char* line, Data* data);

int main (void){
    char* path = "/home/gemini/TUM/CI/CI-Homework_2/Problem_1/testInput10A.txt";
    Data *training[100], *test[100];
    parseFile(path,training,test);
}

void parseFile(char* path, Data** training, Data** test){
    FILE* file;
    int size = 5,i = 0;
    char buff[size];

    file = fopen(path,"r");

    if(file){
        while (fgets(buff, size, (FILE*) file) != NULL) {
            printf("%s",buff);
            //parseTrainingLine(buff, training[i]);
            i++;
        }
    } else{
        printf("Can't open file");
    }
}

void parseTrainingLine(char* line, Data* data){
    data->Input1 = line[0];
    data->Input2 = line[2];
    data->class = line[4];
}