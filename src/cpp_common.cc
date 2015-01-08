#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>
#include "cpp_common.h"
using namespace CppCommonFunction;

int StringFunction::split(const std::string& str, char spliter,std::vector<std::string>& result){
    result.clear();
    std::istringstream ss( str );
    std::string feild;
    while (!ss.eof()){
        std::string x;              // here's a nice, empty string
        std::getline( ss, feild, spliter );  // try to read the next field into it
        result.push_back(feild);
        //   cout << x << endl;    // print it out, even if we already hit EOF
    }
    return result.size();
}
std::string StringFunction::to_string(unsigned int num){
    std::stringstream ss;
    ss<<num;
    return ss.str();
}
std::string StringFunction::join(const std::vector<std::string>& strs,char sep){
    if(strs.size()==0){
        return std::string("");
    }
    if(strs.size()==1){
        return strs[0];
    }
    std::string sep_str(1,sep);
    std::string result;
    for(size_t i=0;i<strs.size()-1;i++){
        result+=strs[i]+sep_str;
    }
    return result+=strs.back();
}

std::string TimeFunction::now(){
    time_t t=std::time(0);
    return std::string(std::asctime(std::localtime(&t)));
}
bool IndexAdapter::InitFromFile(const std::string& file_name,unsigned int index_of_key,char spliter){
    std::ifstream fin(file_name.c_str());
    if(!fin.is_open()){
        std::cerr<<"open file "<<file_name<<" failed."<<std::endl;
        return false;
    }
    keys_.clear();
    std::string line;
    std::vector<std::string> line_vec;
    while(std::getline(fin,line)){
        size_t n=StringFunction::split(line,spliter,line_vec);
        if(index_of_key>=n){
            std::cerr<<"WARNING:broken line,index of key out of line items"<<std::endl;
        }else{
            dict_.insert(std::map<std::string,unsigned int>::value_type(line_vec[index_of_key],keys_.size()));
            keys_.push_back(line_vec[index_of_key]);
        }
    }
    fin.close();
    return true;
}
bool IndexAdapter::GetIndex(const std::string& key,unsigned int& index)const{
    std::map<std::string,unsigned int>::const_iterator iter=dict_.find(key);
    if(iter==dict_.end())   return false;
    index=iter->second;
    return true;
}
bool IndexAdapter::GetValue(unsigned int index,std::string& value)const{
    if(index>=keys_.size())    return false;
    value=keys_[index];
    return true;
}
unsigned int IndexAdapter::GetSize()const{
    return keys_.size();
}

