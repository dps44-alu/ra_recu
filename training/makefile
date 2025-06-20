CXX = g++
CXXFLAGS = -Wall -O2

# Objetivos
all: perceptron backpropagation

perceptron: perceptron.cpp
	$(CXX) $(CXXFLAGS) -o perceptron perceptron.cpp

backpropagation: backpropagation.cpp
	$(CXX) $(CXXFLAGS) -o backpropagation backpropagation.cpp

# Limpieza de archivos generados
clean:
	rm -f perceptron backpropagation
