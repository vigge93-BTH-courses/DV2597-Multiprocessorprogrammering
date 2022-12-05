main: odd gaussseq
oddseq: oddevensortseq.cpp
	g++ -std=c++11 -O2 -o oddevensortseq.exe oddevensortseq.cpp
oddpar: oddevensortpar.cu
	nvcc -o oddevensortpar oddevensortpar.cu
	nvcc -o oddevensortpar2 oddevensortpar2.cu
odd: oddseq oddpar
gaussseq: gaussjordanseq.c
	gcc -O2 -o gaussjordanseq.exe gaussjordanseq.c
r: odd
	./oddevensortpar.exe
	./oddevensortpar2.exe
	./oddevensortseq.exe
clean:
	rm *.exe
	rm *.exp
	rm *.lib