gfortran -c ./fortran/cpsi.f -o ./fortran/cpsi.o
gfortran -c ./fortran/digamma.f90 -o ./fortran/digamma.o
gfortran -c ./quadpack/dqc25c.f -o ./quadpack/dqc25c.o
gfortran -c ./quadpack/dqcheb.f -o ./quadpack/dqcheb.o
gfortran -c ./quadpack/dqk15w.f -o ./quadpack/dqk15w.o
gfortran -c ./quadpack/dqwgtc.f -o ./quadpack/dqwgtc.o
gfortran -c ./quadpack/dqawc.f -o ./quadpack/dqawc.o
gfortran -c ./quadpack/dqawce.f -o ./quadpack/dqawce.o
gfortran -c ./quadpack/dqpsrt.f -o ./quadpack/dqpsrt.o
:: gfortran -c ./fortran/d1mach.f -o ./fortran/d1mach.o
gcc -c ./fortran/d1mach.c -o ./fortran/d1mach.o
gfortran -c ./fortran/pvalint.f90 -o ./fortran/pvalint.o
python setupc.py build_ext --inplace
