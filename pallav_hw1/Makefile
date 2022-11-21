
CC=gcc
CFLAGS += -Wall
AVX_FLAGS= -mavx -DUSE_AVX

OBJ=matsum
OBJ_AVX=$(OBJ)_avx

$(OBJ): $(OBJ).c
	$(CC) $(CFLAGS) -o $(OBJ) $(OBJ).c

$(OBJ_AVX): $(OBJ).c
	$(CC) $(CFLAGS) $(AVX_FLAGS) -o $(OBJ_AVX) $(OBJ).c

all: $(OBJ) $(OBJ_AVX)

clean:
	@rm -f $(OBJ) $(OBJ_AVX)
