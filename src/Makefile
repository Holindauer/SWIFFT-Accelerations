CFLAG = -O2 -march=native -fno-strict-aliasing
CC = gcc
BIN1 = test-amxtile
BIN2 = test-amxtiles-lookup-key
BIN3 = test-amxtile-familiar-lookup
CFILES1 = test-amxtile.c
CFILES2 = test-amxtile-lookup-key.c
CFILES3 = test-amxtile-familiar-lookup.c

all: $(BIN1) $(BIN2) $(BIN3)

$(BIN1): $(CFILES1)
	$(CC) $(CFLAG) $(CFILES1) -o $(BIN1) $(LIBS)

$(BIN2): $(CFILES2)
	$(CC) $(CFLAG) $(CFILES2) -o $(BIN2) $(LIBS)

$(BIN3): $(CFILES3)
	$(CC) $(CFLAG) $(CFILES3) -o $(BIN3) $(LIBS)


clean:
	-rm -f $(BIN1) $(BIN2)

.PHONY: clean
