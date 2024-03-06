SRC = awslib.ml dblib.ml main.ml
LIB = aws,aws.ocsigen,camlp4

LFLAGS = -package $(LIB)
CFLAGS = -linkpkg -thread -syntax camlp4o

all : $(SRC)
	ocamlfind ocamlopt -o main $(CFLAGS) $(LFLAGS) $(SRC)

clean:
	rm -rf _build *.native *.byte *.cmi *.cmx *.o *.out main
