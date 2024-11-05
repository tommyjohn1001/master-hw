CC=clang++
FLAGS= -march=native -std=c++2a
BIN=bin

all: clean server client

server:
	${CC} ${FLAGS} server.cc -o ${BIN}/server

client:
	${CC} ${FLAGS} client.cc -o ${BIN}/client

clean:
	rm -f ${BIN}/*