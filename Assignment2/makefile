main: odd gaussseq
oddseq: oddevensortseq.cpp
	g++ -std=c++11 -O2 -o oddevensortseq.exe oddevensortseq.cpp
oddpar: oddevensortpar.cu
	nvcc -o oddevensortpar oddevensortpar.cu
	nvcc -o oddevensortpar2 oddevensortpar2.cu
odd: oddseq oddpar
gaussseq: gaussjordanseq.c
	gcc -O2 -o gaussjordanseq.exe gaussjordanseq.c
gausspar: gaussjordanpar.cu
	nvcc -lineinfo -o gaussjordanpar gaussjordanpar.cu
gauss: gaussseq gausspar
ro: odd
	./oddevensortpar.exe
	./oddevensortpar2.exe
	./oddevensortseq.exe
rg: gauss
	./gaussjordanseq.exe
	./gaussjordanpar.exe
rgc: gauss
	./gaussjordanseq.exe -P 1 -I fast -n 1024 > seq
	./gaussjordanpar.exe -P 1 -I fast -n 1024 > par
	FC par seq
clean:
	rm *.exe
	rm *.exp
	rm *.lib