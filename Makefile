eft: optlis/static/models/localsearch.c
	mkdir -p build lib
	mkdir -p build/static lib/static
	gcc -c -fPIC optlis/static/models/localsearch.c -o build/static/localsearch.o
	gcc -shared -Wl,-soname,localsearch.so -o lib/static/localsearch.so build/static/localsearch.o

clean:
	rm build/static/localsearch.o

.PHONY: clean
