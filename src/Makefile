CFLAG = -O3 -march=native -fno-strict-aliasing -mamx-tile -mamx-int8 -mavx512f -mavx512dq -msse2 # -O2 default for amx
CC = gcc
BIN1 = test-amx-matmul 
BIN2 = test-amx-fftsum
CFILES1 =test-amx-matmul.c 
CFILES2 =test-amx-fftsum.c amx-fftsum.c

all: $(BIN1) $(BIN2)

$(BIN1): $(CFILES1)
	$(CC) $(CFLAG) $(CFILES1) -o $(BIN1) $(LIBS)

$(BIN2): $(CFILES2)
	$(CC) $(CFLAG) $(CFILES2) -o $(BIN2) $(LIBS)
clean:
	-rm $(BIN1) $(BIN2)

.PHONY: clean