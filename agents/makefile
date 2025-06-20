# Variables
CXX = g++
CXXFLAGS = -c
LDFLAGS = -L../ -lale -Wl,-rpath=../ -lSDL

# Objetivos
TARGETS = manual_agent perceptron_agent backpropagation_agent
OBJS = manual_agent.o perceptron_agent.o backpropagation_agent.o
SRCS = manual_agent.cpp perceptron_agent.cpp backpropagation_agent.cpp 

# Regla por defecto
all: $(TARGETS)

# Compilación de archivos objeto
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Enlazado de los ejecutables
manual_agent: manual_agent.o
	$(CXX) manual_agent.o $(LDFLAGS) -o manual_agent

perceptron_agent: perceptron_agent.o
	$(CXX) perceptron_agent.o $(LDFLAGS) -o perceptron_agent

backpropagation_agent: backpropagation_agent.o
	$(CXX) backpropagation_agent.o $(LDFLAGS) -o backpropagation_agent

# Limpieza de archivos generados
clean:
	rm -f $(OBJS) $(TARGETS)
