CFLAG = -O2 -march=native -fno-strict-aliasing
CC = gcc
BIN1 = test-amxtile
BIN2 = test-amxtiles-lookup-key
CFILES1 = test-amxtile.c
CFILES2 = test-amxtile-lookup-key.c

all: $(BIN1) $(BIN2)

$(BIN1): $(CFILES1)
	$(CC) $(CFLAG) $(CFILES1) -o $(BIN1) $(LIBS)

$(BIN2): $(CFILES2)
	$(CC) $(CFLAG) $(CFILES2) -o $(BIN2) $(LIBS)

clean:
	-rm -f $(BIN1) $(BIN2)

.PHONY: clean
