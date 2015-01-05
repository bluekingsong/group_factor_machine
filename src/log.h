#ifndef _LOG_H_
#define _LOG_H_
#include <iostream>
#include <string>

class Log{
  public:
	static void info(const std::string& tag, const std::string& info){
		println(std::string("INFO"),tag,info);
	}
	static void error(const std::string& tag, const std::string& info){
		println(std::string("ERROR"),tag,info);
	}
	static void warn(const std::string& tag, const std::string& info){
		println(std::string("WARN"),tag,info);
	}
	static void raw(const std::string& info){
		std::cout<<info<<std::endl<<std::flush;
	}
  private:
	static void println(const std::string& type,const std::string& tag, const std::string& info){
		std::cout<<"["<<type<<","<<tag<<"]:"<<info<<std::endl<<std::flush;
	}
};
#endif
