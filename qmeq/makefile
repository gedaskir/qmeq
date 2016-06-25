all: neumannc

neumannc: cpsi.o digamma.o dqc25c.o dqcheb.o dqk15w.o dqwgtc.o dqawc.o dqawce.o dqpsrt.o d1mach.o pvalint.o
	python setupc.py build_ext --inplace

cpsi.o:
	gfortran -c -fPIC ./fortran/cpsi.f -o ./fortran/cpsi.o

digamma.o:
	gfortran -c -fPIC ./fortran/digamma.f90 -o ./fortran/digamma.o

dqc25c.o:
	gfortran -c -fPIC ./quadpack/dqc25c.f -o ./quadpack/dqc25c.o

dqcheb.o:
	gfortran -c -fPIC ./quadpack/dqcheb.f -o ./quadpack/dqcheb.o

dqk15w.o:
	gfortran -c -fPIC ./quadpack/dqk15w.f -o ./quadpack/dqk15w.o

dqwgtc.o:
	gfortran -c -fPIC ./quadpack/dqwgtc.f -o ./quadpack/dqwgtc.o

dqawc.o:
	gfortran -c -fPIC ./quadpack/dqawc.f -o ./quadpack/dqawc.o

dqawce.o:
	gfortran -c -fPIC ./quadpack/dqawce.f -o ./quadpack/dqawce.o

dqpsrt.o:
	gfortran -c -fPIC ./quadpack/dqpsrt.f -o ./quadpack/dqpsrt.o

d1mach.o:
	gcc -c -fPIC ./fortran/d1mach.c -o ./fortran/d1mach.o

pvalint.o:
	gfortran -c -fPIC ./fortran/pvalint.f90 -o ./fortran/pvalint.o