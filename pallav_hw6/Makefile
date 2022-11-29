CC=nvcc
CFLAGS += -O2 -g

OBJ=conv

$(OBJ): $(OBJ).cu
	$(CC) $(CFLAGS) -o $(OBJ) $(OBJ).cu

all: $(OBJ)

clean:
	@rm -f $(OBJ)
