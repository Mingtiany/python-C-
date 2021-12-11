#include <Python.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <time.h>

#define NUM 3
#define THREAD_NUM 1
using namespace std;

std::mutex mut;

PyObject* pMod = NULL;
PyObject* pFunc = NULL;

clock_t time_each_thread = 0;
int func(int idx,float *data) {
	cout << "This is sub thread " << idx << endl;
	int predictNum = 510;
	clock_t thread_s = clock();
	for (int preN = 0; preN < predictNum; preN++) {
		PyObject* pParm = NULL;
		PyObject* pArgs = NULL;
		PyObject* pRetVal = NULL;
		float iRetVal[NUM] = { 0 };

		////////read data//////////
		// copying the data to the list
		int j = 0;
		pArgs = PyList_New(NUM); // 新建一个列表，并填入数据
		while (j < NUM)
		{
			PyList_SET_ITEM(pArgs, j, Py_BuildValue("f", data[j]));
			j++;
		}
		////////TEST//////////
		pParm = PyTuple_New(1);//新建一个元组，参数只能通过元组传入python程序
		PyTuple_SetItem(pParm, 0, pArgs);//传入的参数，data_list
		std::unique_lock<std::mutex> lock(mut);
		pRetVal = PyEval_CallObject(pFunc, pParm);//这里开始执行py脚本
												  //解析结果
												  //int list_len = PyList_Size(pRetVal);
		int list_len = PyObject_Size(pRetVal);
		//cout << "sub thread " << idx << ": list_len = " << list_len << endl;
		PyObject *list_item = NULL;
		if (list_len != NUM) {
			std::cout << "sub thread " << idx << " :list_len != NUM  " << "list_len = " << list_len << std::endl;
			return -3;
		}
		for (int i = 0; i < list_len; i++) {
			list_item = PyList_GetItem(pRetVal, i);
			PyArg_Parse(list_item, "f", &iRetVal[i]);//解析list_item到iRetVal
		}
		//cout predict value
		/*
		cout << "sub thread " << idx << ": predict value:";
		for (int i = 0; i < list_len; i++) {
			cout << iRetVal[i] << " ";
		}
		cout << endl;
		*/
	}
	clock_t thread_e = clock();
	time_each_thread += (thread_e - thread_s);
	return 0;
}

int initPy() {
	////////initial//////////
	Py_Initialize();
	if (!Py_IsInitialized()) {
		cout << "error:Py_IsInitialized" << endl;
		return -1;
	}
	// Path of the python file. 需要更改为 python 文件所在路径
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('H:/vvencDLv2/deepLearning/simpleTest/')");

	const  char* modulName = "simplePredict";    //这个是被调用的py文件模块名字
	pMod = PyImport_ImportModule(modulName);
	if (!pMod)
	{
		cout << "error:PyImport_ImportModule" << endl;
		return -1;
	}

	const char* funcName = "predict";  //这是此py文件模块中被调用的函数名字
	pFunc = PyObject_GetAttrString(pMod, funcName);
	if (!pFunc)
	{
		cout << "error:PyObject_GetAttrString" << endl;
		return -2;
	}
	return 0;
}

int destroyPy() {
	printf("Done!\n");
	Py_Finalize();
	return 0;
}

int main(int argc, char** argv)
{

	//init python
	initPy();
	
	float data[THREAD_NUM][NUM];
   //init data
	for (int i = 0; i < THREAD_NUM;i ++) {
		for (int j = 0; j < NUM;j++) {
			data[i][j] = i * NUM + j;
		}
	}

	//start sub thread
	thread * t[THREAD_NUM];
	for (int thread_idx = 0; thread_idx < THREAD_NUM; thread_idx++) {
		//std::this_thread::sleep_for(10ms);
		t[thread_idx] = new thread(func, thread_idx, data[thread_idx]);
	}

	for (int thread_idx = 0; thread_idx < THREAD_NUM; thread_idx++) {
		t[thread_idx]->join();
		cout << "thread " << thread_idx << " end" << endl;
	}

	cout << "time_each_thread: " << time_each_thread  <<"ms"<< endl;

	//destroy python 
	destroyPy();

	return 0;
}