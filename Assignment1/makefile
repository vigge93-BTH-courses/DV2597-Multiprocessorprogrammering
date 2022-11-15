main:
	make threads gauss qsort
threads:
	gcc -O2 -o pthreadcreate.exe pthreadcreate.c -lpthread
	gcc -O2 -o pthreadcreate2.exe pthreadcreate2.c -lpthread
	gcc -O2 -o pthreadcreate3.exe pthreadcreate3.c -lpthread
	gcc -O2 -o bankaccount.exe bankaccount.c -lpthread
	gcc -O2 -o matmulseq.exe matmulseq.c -lpthread
	gcc -o false_sharing.exe false_sharing.c -lpthread
gauss:
	gcc -O2 -o gaussianseq.exe gaussianseq.c -lpthread
	gcc -O2 -o gaussianpar.exe gaussianpar.c -lpthread
	gcc -O2 -o gaussianpar2.exe gaussianpar2.c -lpthread
	gcc -O2 -o gaussianpar3.exe gaussianpar3.c -lpthread
qsort:
	gcc -O2 -o qsortseq.exe qsortseq.c -lpthread
	gcc -O2 -o qsortpar1.exe qsortpar1.c -lpthread
	gcc -O2 -o qsortpar3.exe qsortpar3.c -lpthread

clean:
	rm *.exe
