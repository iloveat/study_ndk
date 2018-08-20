SOURCES += validation.cpp


CXX = arm-linux-androideabi-clang++


XXFLAGS += -std=c++11 -mhard-float -ldl -lz -llog -O3 -Wl,--no-warn-mismatch -lm_hard


libvalidation.so : 
	$(CXX) -shared -o libvalidation.so $(SOURCES) $(XXFLAGS)


clean :
	rm -f *.so


.PHONY : clean


