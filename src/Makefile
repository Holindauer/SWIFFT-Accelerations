CFLAG = -O3 -march=native -fno-strict-aliasing -mamx-tile -mamx-int8 -mavx512f -mavx512dq # -O2 default for amx
CC = gcc
BIN1 = test-amx-matmul 
CFILES1 =test-amx-matmul.c 

all: $(BIN1)

$(BIN1): $(CFILES1)
	$(CC) $(CFLAG) $(CFILES1) -o $(BIN1) $(LIBS)

clean:
	-rm $(BIN1) 

.PHONY: clean