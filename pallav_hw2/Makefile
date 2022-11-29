CC=nvcc
CFLAGS += -O2 -g

DIVOBJ=PPdivergent

$(DIVOBJ): $(DIVOBJ).cu
	$(CC) $(CFLAGS) -o $(DIVOBJ) $(DIVOBJ).cu

all: $(DIVOBJ)

clean:
	@rm -f $(DIVOBJ)
