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
		pArgs = PyList_New(NUM); // �½�һ���б�����������
		while (j < NUM)
		{
			PyList_SET_ITEM(pArgs, j, Py_BuildValue("f", data[j]));
			j++;
		}
		////////TEST//////////
		pParm = PyTuple_New(1);//�½�һ��Ԫ�飬����ֻ��ͨ��Ԫ�鴫��python����
		PyTuple_SetItem(pParm, 0, pArgs);//����Ĳ�����data_list
		std::unique_lock<std::mutex> lock(mut);
		pRetVal = PyEval_CallObject(pFunc, pParm);//���￪ʼִ��py�ű�
												  //�������
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
			PyArg_Parse(list_item, "f", &iRetVal[i]);//����list_item��iRetVal
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
	// Path of the python file. ��Ҫ����Ϊ python �ļ�����·��
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('H:/vvencDLv2/deepLearning/simpleTest/')");

	const  char* modulName = "simplePredict";    //����Ǳ����õ�py�ļ�ģ������
	pMod = PyImport_ImportModule(modulName);
	if (!pMod)
	{
		cout << "error:PyImport_ImportModule" << endl;
		return -1;
	}

	const char* funcName = "predict";  //���Ǵ�py�ļ�ģ���б����õĺ�������
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