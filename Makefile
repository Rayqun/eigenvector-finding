##########################################################
###     THIS FILE WILL BE OVERWRITTEN WHEN GRADING.    ###   
###              MODIFY IT AT YOUR OWN RISK            ###
##########################################################
HOME = /home/ff/cs61c
CC = gcc -std=gnu99 

OMP = -fopenmp
LIBS = -lpthread -lm
# a pretty good flag selection for this machine...
CFLAGS = -fopenmp -O2 -pipe -fno-omit-frame-pointer -march=native -g

all:	bench-naive bench-fast

bench-naive: benchmark.o eig-naive.o
	$(CC) -o $@ benchmark.o eig-naive.o $(LIBS)

bench-fast: benchmark.o eig-fast.o
	$(CC) -o $@ $(OMP) benchmark.o eig-fast.o $(LIBS)

%.o: %.c
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f *~ bench-fast bench-naive *.o
