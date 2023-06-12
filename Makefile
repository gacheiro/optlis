static: optlis/static/models/localsearch.c
	mkdir -p build lib
	mkdir -p build/static lib/static
	gcc -c -fPIC optlis/static/models/localsearch.c -o build/static/localsearch.o
	gcc -shared -Wl,-soname,localsearch.so -o lib/static/localsearch.so build/static/localsearch.o

dynamic: optlis/dynamic/models/localsearch.c
	mkdir -p build lib
	mkdir -p build/dynamic lib/dynamic
	gcc -c -fPIC optlis/dynamic/models/localsearch.c -o build/dynamic/localsearch.o
	gcc -shared -Wl,-soname,localsearch.so -o lib/dynamic/localsearch.so build/dynamic/localsearch.o

clean:
	rm build/static/localsearch.o
	rm build/dynamic/localsearch.o

.PHONY: clean
