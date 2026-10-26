main: main.cu
	nvcc -arch=sm_86 -o $@ $^ 

run_main: main
	./main