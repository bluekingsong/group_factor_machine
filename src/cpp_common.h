#ifndef CPP_COMMON_COMMON_H_
#define CPP_COMMON_COMMON_H_
#include <vector>
#include <string>
#include <map>
namespace CppCommonFunction{

class StringFunction{
  public:
    static int split(const std::string& str, char spliter,std::vector<std::string>& result);
    static std::string join(const std::vector<std::string>& strs,char sep);
    static std::string to_string(unsigned int);
};
class TimeFunction{
  public:
    static std::string now();
};
class IndexAdapter{
  public:
      IndexAdapter(){
      }
      bool InitFromFile(const std::string& file_name,unsigned int index_of_key=0,char spliter='\t');
      bool GetIndex(const std::string& key,unsigned int& index)const;
      bool GetValue(unsigned int index, std::string& value)const;
      unsigned int GetSize()const;
  private:
      std::vector<std::string> keys_;
      std::map<std::string,unsigned int> dict_;
};

}; //CppCommonFunction
#endif // CPP_COMMON_COMMON_H_
