main: main.cu
	nvcc -gencode arch=compute_75,code=sm_75 \
	     -gencode arch=compute_86,code=sm_86 \
	     -o $@ $^

run: main
	./main $(SIZE)