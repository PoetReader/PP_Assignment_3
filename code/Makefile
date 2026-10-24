main: main.cu
	nvcc -arch=sm_50 -o $@ $^ 

serial: serial.c
	gcc -o $@ $^ -lm

run: serial
	./serial
