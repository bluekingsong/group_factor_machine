#include <cstdio>
#include <math.h>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cerrno>
#include "dataset.h"
#include "log.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using namespace std;

/** copy from liblinear source code **/

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input){
	int len;
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;
	while(strrchr(line,'\n') == NULL){
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}
void exit_input_error(int i){
    std::cerr<<"exit on "<<i<<std::endl;
	exit(i);
}

struct FeatureNode *x_space;

// read in a problem (in libsvm format)
Problem read_problem(const char *filename){
	Problem prob;
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	if(fp == NULL){
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL){
		char *p = strtok(line," \t"); // label
		// features
		while(1){
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
        elements -= 3; // group factor machine data format
		prob.l++;
		++elements; // for the terminate node of a instance
	}
    std::cerr<<"num of instance="<<prob.l<<" number of elments="<<elements<<std::endl;
	rewind(fp);
	prob.y = Malloc(uint8_t,prob.l);
    prob.weight = Malloc(float,prob.l);
	prob.x = Malloc(struct FeatureNode*,prob.l);
	x_space = Malloc(struct FeatureNode,elements);
	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++){
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		//if(i>190834) printf("line=%s\n",line);
		prob.x[i] = &x_space[j];
        char *pvid = strtok(line," \t");
        char *weightStr = strtok(NULL," \t");
        char *posStr = strtok(NULL," \t");
		//label = strtok(line," \t\n");
		label = strtok(NULL," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);
		prob.y[i] = atoi(label);
        double weight = 1;
        if(label)    weight = atof(weightStr);
        prob.weight[i] = weight;
		while(1){
            char *groupStr = strtok(NULL,":");
			if(groupStr == NULL)		break;
            x_space[j].group = atoi(groupStr);
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			errno = 0;
            int tmp = (int) strtol(idx,&endptr,10);
			x_space[j].index = tmp;
			if(endptr == idx || errno != 0 || *endptr != '\0') // || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else{
				inst_max_index = x_space[j].index;
			}
			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
            //std::cerr<<"read "<<x_space[j].group<<":"<<idx<<":"<<val<<std::endl;
			++j;
		}
		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}
	prob.n = max_index;
	fclose(fp);
	free(line);
	return prob;
}
void Problem::list_problem_struct(const Problem& prob){
	printf("number of instances=%d\n",prob.l);
	printf("number of features=%d\n",prob.n);
	printf("the 5th line of data file is:\n");
	FeatureNode *x = prob.x[4];
	while (x->index >= 0){
		printf("%d:%d:%d ",x->group,x->index,int(x->value));
		x += 1;
	}
	printf("\n");
}

Problem Problem::unittest(const std::string& filename){
	Log::raw("===========DataSet::unittest===============");
	Problem prob;
	int l = 10;
	prob.n = 5; prob.l = l;
	int label[] = {1,1,0,0,0,0,0,0,0,0};
	int indies[][2]= { {1,2},
			{2,3},
			{3,4},
			{4,5},
			{5,1},
			{1,3},
			{1,4},
			{1,5},
			{2,4},
			{2,5}
	};
	FeatureNode *nodes = new FeatureNode[3 * l];
	prob.x = new FeatureNode* [l];
	prob.y = new uint8_t[l];
	for(int i = 0; i < l; ++i){
		prob.y[i] = label[i];
		prob.x[i] = nodes + 3 * i;
		for(int j = 0; j < 2; ++j){
			prob.x[i][j].index = indies[i][j]-1;
			prob.x[i][j].value = 1;
		}
		prob.x[i][2].index = -1;
	}

	printf("number of instances=%d\n",prob.l);
	printf("number of features=%d\n",prob.n);
	printf("the 5th line of data file is:\n");
		FeatureNode *x = prob.x[4];
		while (x->index >= 0){
			printf("%d:%f ",x->index,x->value);
			x += 1;
		}
	printf("\n");
	printf("the 10th line of data file is:\n");
	x = prob.x[9];
	while (x->index >= 0){
		printf("%d:%f ",x->index,x->value);
		x += 1;
	}
	printf("\n");
	Log::raw("+++++++++++DataSet::unittest+++++++++++++++");
	return prob;
}
