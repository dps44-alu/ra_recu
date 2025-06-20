#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <map>

using namespace std;

// Network architecture
const int INPUT_SIZE = 26;
const int HIDDEN_SIZE = 32;  // Reduced for stability
const int OUTPUT_SIZE = 3;

// Training parameters
const double LEARNING_RATE = 0.0001;  // Reduced learning rate
const double MIN_LEARNING_RATE = 0.000001;
const int MAX_EPOCHS = 2000;  // Increased max epochs
const double MOMENTUM = 0.5;  // Reduced momentum
const int BATCH_SIZE = 64;    // Increased batch size
const double EPSILON = 1e-7;  // Small constant to prevent division by zero

struct Network {
    vector<vector<vector<double>>> weights;
    vector<vector<double>> biases;
    vector<vector<vector<double>>> momentum_weights;
    vector<vector<double>> momentum_biases;
};

struct TrainingData {
    vector<double> inputs;
    vector<double> targets;
};

// Safe math operations
double safe_exp(double x) {
    if (x > 88.0) return exp(88.0);
    if (x < -88.0) return exp(-88.0);
    return exp(x);
}

double safe_log(double x) {
    return log(max(x, EPSILON));
}

// Initialize network with careful weight initialization
Network initializeNetwork() {
    Network net;
    random_device rd;
    mt19937 gen(rd());
    
    // Use He initialization
    double scale_hidden = sqrt(2.0 / INPUT_SIZE);
    double scale_output = sqrt(2.0 / HIDDEN_SIZE);
    
    net.weights.resize(2);
    net.biases.resize(2);
    net.momentum_weights.resize(2);
    net.momentum_biases.resize(2);
    
    // Input to hidden layer
    uniform_real_distribution<> dis_hidden(-scale_hidden, scale_hidden);
    net.weights[0].resize(HIDDEN_SIZE, vector<double>(INPUT_SIZE));
    net.biases[0].resize(HIDDEN_SIZE);
    net.momentum_weights[0].resize(HIDDEN_SIZE, vector<double>(INPUT_SIZE, 0.0));
    net.momentum_biases[0].resize(HIDDEN_SIZE);
    
    for(auto& neuron : net.weights[0]) {
        for(double& weight : neuron) {
            weight = dis_hidden(gen) * 0.1; // Scale down initial weights
        }
    }
    
    // Hidden to output layer
    uniform_real_distribution<> dis_output(-scale_output, scale_output);
    net.weights[1].resize(OUTPUT_SIZE, vector<double>(HIDDEN_SIZE));
    net.biases[1].resize(OUTPUT_SIZE);
    net.momentum_weights[1].resize(OUTPUT_SIZE, vector<double>(HIDDEN_SIZE, 0.0));
    net.momentum_biases[1].resize(OUTPUT_SIZE);
    
    for(auto& neuron : net.weights[1]) {
        for(double& weight : neuron) {
            weight = dis_output(gen) * 0.1; // Scale down initial weights
        }
    }
    
    return net;
}

// ReLU with gradient clipping
double relu(double x) {
    return min(max(x, 0.0), 5.0); // Clip activation
}

double relu_derivative(double x) {
    return (x > 0.0 && x < 5.0) ? 1.0 : 0.0;
}

// Forward pass with numerical stability checks
vector<vector<double>> forward(const Network& net, const vector<double>& input, bool training = false) {
    vector<vector<double>> activations(2);
    
    // Hidden layer
    activations[0].resize(HIDDEN_SIZE);
    for(int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = net.biases[0][i];
        for(int j = 0; j < INPUT_SIZE; j++) {
            sum += net.weights[0][i][j] * input[j];
        }
        // Clip sum to prevent explosion
        sum = max(min(sum, 100.0), -100.0);
        activations[0][i] = relu(sum);
    }
    
    // Output layer with stable softmax
    activations[1].resize(OUTPUT_SIZE);
    double max_val = -numeric_limits<double>::infinity();
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = net.biases[1][i];
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net.weights[1][i][j] * activations[0][j];
        }
        // Clip sum to prevent explosion
        sum = max(min(sum, 100.0), -100.0);
        activations[1][i] = sum;
        max_val = max(max_val, sum);
    }
    
    // Stable softmax implementation
    double sum_exp = 0.0;
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        activations[1][i] = safe_exp(activations[1][i] - max_val);
        sum_exp += activations[1][i];
    }
    
    sum_exp = max(sum_exp, EPSILON);
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        activations[1][i] /= sum_exp;
        // Ensure outputs are valid probabilities
        activations[1][i] = max(min(activations[1][i], 1.0 - EPSILON), EPSILON);
    }
    
    return activations;
}

// Stable backpropagation
void backpropagate(Network& net, const vector<double>& input, const vector<double>& target, 
                  const vector<vector<double>>& activations, double learning_rate) {
    // Output layer deltas with gradient clipping
    vector<double> output_deltas(OUTPUT_SIZE);
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        output_deltas[i] = activations[1][i] - target[i];
        output_deltas[i] = max(min(output_deltas[i], 1.0), -1.0); // Clip gradients
    }
    
    // Hidden layer deltas
    vector<double> hidden_deltas(HIDDEN_SIZE);
    for(int i = 0; i < HIDDEN_SIZE; i++) {
        double error = 0.0;
        for(int j = 0; j < OUTPUT_SIZE; j++) {
            error += output_deltas[j] * net.weights[1][j][i];
        }
        hidden_deltas[i] = error * relu_derivative(activations[0][i]);
        hidden_deltas[i] = max(min(hidden_deltas[i], 1.0), -1.0); // Clip gradients
    }
    
    // Update weights with careful learning rate and momentum
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            double delta = output_deltas[i] * activations[0][j];
            delta = max(min(delta * learning_rate, 0.1), -0.1); // Clip update
            net.momentum_weights[1][i][j] = MOMENTUM * net.momentum_weights[1][i][j] + delta;
            net.weights[1][i][j] -= net.momentum_weights[1][i][j];
        }
        net.momentum_biases[1][i] = MOMENTUM * net.momentum_biases[1][i] + 
                                   learning_rate * output_deltas[i];
        net.biases[1][i] -= net.momentum_biases[1][i];
    }
    
    for(int i = 0; i < HIDDEN_SIZE; i++) {
        for(int j = 0; j < INPUT_SIZE; j++) {
            double delta = hidden_deltas[i] * input[j];
            delta = max(min(delta * learning_rate, 0.1), -0.1); // Clip update
            net.momentum_weights[0][i][j] = MOMENTUM * net.momentum_weights[0][i][j] + delta;
            net.weights[0][i][j] -= net.momentum_weights[0][i][j];
        }
        net.momentum_biases[0][i] = MOMENTUM * net.momentum_biases[0][i] + 
                                   learning_rate * hidden_deltas[i];
        net.biases[0][i] -= net.momentum_biases[0][i];
    }
}

vector<TrainingData> loadTrainingData(const string& filename) {
    vector<TrainingData> data;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Could not open file: " << filename << endl;
        return data;
    }
    
    string line;
    int line_count = 0;
    map<int, int> action_mapping;  // Para contar las acciones
    
    while (getline(file, line)) {
        line_count++;
        if (line.rfind("Step", 0) == 0) {
            istringstream step_line(line);
            string step_word, colon;
            int action;
            
            // Parse "Step", action, and colon
            step_line >> step_word >> action >> colon;
            
            // Mapear las acciones 0,1,2,3 a 0,1,2 (combinando algunas)
            int mapped_action = action;
            if (action == 3) mapped_action = 2;  // Mapeamos 3 a 2
            if (mapped_action >= 0 && mapped_action <= 2) {
                vector<double> targets(3, 0.0);
                targets[mapped_action] = 1.0;
                
                // Initialize inputs with default values
                vector<double> inputs(INPUT_SIZE, 0.0);
                
                // Parse RAM values
                string ram_entry;
                while (step_line >> ram_entry) {
                    if (ram_entry[0] == 'R') {
                        size_t underscore_pos = ram_entry.find('_');
                        if (underscore_pos != string::npos) {
                            try {
                                int ram_pos = stoi(ram_entry.substr(1, underscore_pos - 1));
                                double ram_val = stod(ram_entry.substr(underscore_pos + 1));
                                
                                if (ram_pos >= 0 && ram_pos < INPUT_SIZE) {
                                    // Normalize input to [-1, 1]
                                    inputs[ram_pos] = (ram_val / 255.0) * 2.0 - 1.0;
                                }
                            } catch (const exception& e) {
                                // Silently skip invalid entries
                                continue;
                            }
                        }
                    }
                }
                
                data.push_back({inputs, targets});
                action_mapping[action]++;
            } else {
                cerr << "Invalid action value: " << action << " at line " << line_count << endl;
            }
        }
    }
    
    // Imprimir estadísticas de las acciones
    cout << "\nAction distribution in training data:" << endl;
    for (const auto& pair : action_mapping) {
        cout << "Action " << pair.first << ": " << pair.second << " samples (" 
             << (pair.second * 100.0 / data.size()) << "%)" << endl;
    }
    cout << endl;
    
    file.close();
    return data;
}
// Print weights in the game agent format
void printWeights(const Network& net) {
    cout << "double s_w[s_numIAs][3][s_kinputs] = {\n { ";
    
    // Calculate combined weights for each output
    for(int output = 0; output < OUTPUT_SIZE; output++) {
        cout << "{ " << fixed << setprecision(2) << net.biases[1][output];
        
        // Combine weights through hidden layer
        for(int input = 0; input < INPUT_SIZE - 1; input++) {
            double combined_weight = 0.0;
            for(int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
                combined_weight += net.weights[0][hidden][input] * net.weights[1][output][hidden];
            }
            cout << " , " << combined_weight;
        }
        
        cout << "}";
        if(output < OUTPUT_SIZE - 1) cout << "\n, ";
    }
    cout << "\n}\n};" << endl;
}
int main() {
    Network net = initializeNetwork();
    vector<TrainingData> training_data = loadTrainingData("../agents/datos.csv");
    cout << "Loaded " << training_data.size() << " training samples" << endl;
    
    if(training_data.empty()) {
        cerr << "No training data found!" << endl;
        return 1;
    }
    
    // Print initial distribution of actions
    vector<int> initial_actions(3, 0);
    for(const auto& sample : training_data) {
        int action = max_element(sample.targets.begin(), sample.targets.end()) - sample.targets.begin();
        initial_actions[action]++;
    }
    cout << "Initial action distribution: [";
    for(int i = 0; i < 3; i++) {
        cout << (initial_actions[i] * 100.0 / training_data.size()) << "% ";
    }
    cout << "]" << endl;
    
    // Normalize inputs to [-1, 1] range
    for(auto& sample : training_data) {
        for(double& input : sample.inputs) {
            input = max(min(input, 1.0), -1.0);
        }
    }
    
    random_device rd;
    mt19937 gen(rd());
    
    double best_accuracy = 0.0;
    int epochs_without_improvement = 0;
    double learning_rate = LEARNING_RATE;
    
    for(int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        shuffle(training_data.begin(), training_data.end(), gen);
        
        double total_error = 0.0;
        int correct_predictions = 0;
        
        for(size_t i = 0; i < training_data.size(); i += BATCH_SIZE) {
            size_t batch_end = min(i + BATCH_SIZE, training_data.size());
            
            for(size_t j = i; j < batch_end; j++) {
                vector<vector<double>> activations = forward(net, training_data[j].inputs, true);
                
                double sample_error = 0.0;
                for(int k = 0; k < OUTPUT_SIZE; k++) {
                    sample_error += pow(training_data[j].targets[k] - activations[1][k], 2);
                }
                total_error += sample_error;
                
                int predicted = max_element(activations[1].begin(), activations[1].end()) - 
                              activations[1].begin();
                int actual = max_element(training_data[j].targets.begin(), 
                                      training_data[j].targets.end()) - 
                           training_data[j].targets.begin();
                
                if(predicted == actual) correct_predictions++;
                
                backpropagate(net, training_data[j].inputs, training_data[j].targets, 
                            activations, learning_rate);
            }
        }
        
        double accuracy = static_cast<double>(correct_predictions) / training_data.size();
        double avg_error = total_error / training_data.size();
        
        if(epoch % 10 == 0) {
            cout << "Epoch " << epoch << 
                    " - Error: " << avg_error << 
                    " - Accuracy: " << (accuracy * 100) << "%" <<
                    " - Learning Rate: " << learning_rate;
            
            // Print distribution of predictions
            vector<int> action_counts(3, 0);
            for(const auto& sample : training_data) {
                vector<vector<double>> pred = forward(net, sample.inputs);
                int predicted = max_element(pred[1].begin(), pred[1].end()) - pred[1].begin();
                action_counts[predicted]++;
            }
            cout << " - Action distribution: [";
            for(int i = 0; i < 3; i++) {
                cout << (action_counts[i] * 100.0 / training_data.size()) << "% ";
            }
            cout << "]" << endl;
        }
        
        if(accuracy > best_accuracy) {
            best_accuracy = accuracy;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement++;
            if(epochs_without_improvement % 10 == 0 && learning_rate > MIN_LEARNING_RATE) {
                learning_rate *= 0.5;
                cout << "Reducing learning rate to " << learning_rate << endl;
            }
            if(epochs_without_improvement >= 30) {
                cout << "\nEarly stopping triggered after " << epoch << " epochs" << endl;
                cout << "Best accuracy achieved: " << (best_accuracy * 100) << "%" << endl;
                break;
            }
        }
    }
    
    cout << "\nTrained weights for game agent:" << endl;
    printWeights(net);
    
    return 0;
}