main: main.cu
	nvcc -arch=sm_86 -o $@ $^ 

run: main
	./main