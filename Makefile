
#MatrixMultiply

SIMULATOR = 0
MEASURE = 0

ifeq ($(SIMULATOR), 1)
	OPENCL_DIR = /home/stardica/Dropbox/CDA7919DoctoralResearch/runtime
	OPENCL_INC = $(OPENCL_DIR)/src/runtime/include/
	OPENCL_LIB = $(OPENCL_DIR)/Release_Static/ -lm2s-opencl
	KERNEL_PATH_VAR='"/home/stardica/Dropbox/CDA7919DoctoralResearch/Rodinia_Benchmarks/OpenCL/backprop/backprop_kernel.cl.bin.GPU"'
	CC_FLAGS = -g -O2 -m32 -Wall -Werror
	CC = gcc -DKERNEL_PATH=$(KERNEL_PATH_VAR) -DM2S_CGM_OCL_SIM=$(SIMULATOR) -DM2S_CGM_OCL_MEASURE=$(MEASURE)
	LINKER_FLAGS= -static -pthread -ldl -lrt
	DO_COMPILE = $(CC) $(CC_FLAGS) backprop_ocl.cpp backprop.o imagenet.o facetrain.o -o backprop_ocl -I$(OPENCL_INC) -L$(OPENCL_LIB) $(LINKER_FLAGS)
else
	# this is for the amdapp sdk.
	OPENCL_DIR = /opt/AMDAPP
	OPENCL_INC = $(OPENCL_DIR)/include/
	OPENCL_LIB = $(OPENCL_DIR)/lib/x86_64/ -lOpenCL
	KERNEL_PATH_VAR='"/home/stardica/Dropbox/CDA7919DoctoralResearch/MatrixMultiply/MatrixMultiply.cl.bin.GPU"'
	CC_FLAGS = -g -O2 -Wall -Werror
	CC = gcc -DKERNEL_PATH=$(KERNEL_PATH_VAR) -DM2S_CGM_OCL_SIM=$(SIMULATOR) -DM2S_CGM_OCL_MEASURE=$(MEASURE)
	LINKER_FLAGS= -pthread	 
endif

all: MM64

MM64:
	$(CC) $(CC_FLAGS) MatrixMultiply.c -o MatrixMultiply -I$(OPENCL_INC) -L$(OPENCL_LIB) $(LINKER_FLAGS)

#MM32:
#	$(CC) $(CC_FLAGS) MatrixMultiply.o -o MatrixMultiply -I$(OPENCL_INC) -L$(OPENCL_LIB) $(LINKER_FLAGS)
	
clean:
	rm -f *.o *~ backprop_ocl *.linkinfo *.out
