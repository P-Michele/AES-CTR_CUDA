NVCC = nvcc
NVCCFLAGS = -O2

main: main.o aes.o
	$(NVCC) $(NVCCFLAGS) -o main main.o aes.o

main.o: main.cu aes.cuh
	$(NVCC) $(NVCCFLAGS) -c main.cu

aes.o: aes.cu aes.cuh
	$(NVCC) $(NVCCFLAGS) -c aes.cu

clean:
	rm -f *.o main