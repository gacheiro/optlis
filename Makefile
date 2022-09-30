eft: optlis/solvers/localsearch.c
	mkdir -p build lib
	gcc -c -fPIC optlis/solvers/localsearch.c -o build/localsearch.o
	gcc -shared -Wl,-soname,localsearch.so -o lib/localsearch.so build/localsearch.o

clean:
	rm build/localsearch.o

.PHONY: clean
