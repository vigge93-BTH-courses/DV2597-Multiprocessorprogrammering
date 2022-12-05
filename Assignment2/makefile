main:
	make odd gauss
odd:
	g++ -std=c++11 -O2 -o oddevensortseq.exe oddevensortseq.cpp
	nvcc -o oddevensortpar oddevensortpar.cu
gauss:
	gcc -O2 -o gaussjordanseq.exe gaussjordanseq.c
clean:
	rm *.exe