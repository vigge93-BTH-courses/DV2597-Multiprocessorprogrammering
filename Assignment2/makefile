main: oddeven gauss
oddeven: oddevenseq oddevenpar
gauss: gaussseq gausspar

oddevenseq: oddevensortseq.exe
oddevensortseq.exe: oddevensortseq.cpp
	g++ -std=c++11 -O2 -o oddevensortseq.exe oddevensortseq.cpp
oddevenpar: oddevensortpar_v1.exe oddevensortpar_v2.exe
oddevensortpar_v1.exe: oddevensortpar_v1.cu
	nvcc -o oddevensortpar_v1 oddevensortpar_v1.cu
oddevensortpar_v2.exe: oddevensortpar_v2.cu
	nvcc -o oddevensortpar_v2 oddevensortpar_v2.cu

gaussseq: gaussjordanseq.exe
gaussjordanseq.exe: gaussjordanseq.cpp
	g++ --std=c++11 -O2 -o gaussjordanseq.exe gaussjordanseq.cpp
gausspar: gaussjordanpar_v1.exe gaussjordanpar_v2.exe gaussjordanpar_v3.exe
gaussjordanpar_v1.exe: gaussjordanpar_v1.cu
	nvcc -lineinfo -o gaussjordanpar_v1 gaussjordanpar_v1.cu
gaussjordanpar_v2.exe: gaussjordanpar_v2.cu
	nvcc -lineinfo -o gaussjordanpar_v2 gaussjordanpar_v2.cu
gaussjordanpar_v3.exe: gaussjordanpar_v3.cu
	nvcc -lineinfo -o gaussjordanpar_v3 gaussjordanpar_v3.cu

run_oddeven: oddeven
	./oddevensortseq.exe
	./oddevensortpar_v1.exe
	./oddevensortpar_v2.exe
run_gauss: gauss
	./gaussjordanseq.exe
	./gaussjordanpar_v1.exe
	./gaussjordanpar_v2.exe
	./gaussjordanpar_v3.exe
run_both: oddeven gauss run_oddeven run_gauss
run_compare_gauss: gauss
	./gaussjordanseq.exe -P 1 > seq
	./gaussjordanpar_v3.exe -P 1 > par
	FC par seq
run_benchmark: oddevensortseq.exe oddevensortpar_v1.exe oddevensortpar_v2.exe gaussjordanseq.exe gaussjordanpar_v3.exe
	for number in 1 2 3 4 5; do \
		echo $$number; \
		./oddevensortseq.exe >> oddevenseq ; \
		echo Oddevenseq done; \
		./oddevensortpar_v1.exe >> oddevenpar_v1 ;\
		echo Oddevenpar1 done; \
		./oddevensortpar_v2.exe >> oddevenpar_v2 ;\
		echo Oddevenpar2 done; \
		./gaussjordanseq.exe >> gaussseq ; \
		echo gaussseq done; \
		./gaussjordanpar_v3.exe >> gausspar ; \
		echo gausspar done; \
	done

clean_bench:
	rm par seq oddevenseq oddevenpar gaussseq gausspar
clean:
	rm *.exe *.exp *.lib