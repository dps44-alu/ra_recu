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
#include <unordered_map>

using namespace std;

// Network architecture (must match the agent)
const std::vector<int> RAM_POSITIONS = {
    0x10, 0x30, 0x40, 0x21, 0x31, 0x22, 0x32, 0x42, 0x13, 0x44,
    0x25, 0x45, 0x55, 0x16, 0x26, 0x46, 0x07, 0x47, 0x08, 0x09,
    0x49, 0x4a, 0x4b, 0x0f, 0x1f, 0x2f
};
const int INPUT_SIZE  = RAM_POSITIONS.size();
const int HIDDEN_SIZE = 30;    // Same as the game agent
const int OUTPUT_SIZE = 3;

// Training parameters
const double LEARNING_RATE     = 0.0001;  // Initial learning rate
const double MIN_LEARNING_RATE = 0.000001;
const int    MAX_EPOCHS        = 2000;
const double MOMENTUM          = 0.5;
const int    BATCH_SIZE        = 64;

// Regularization and validation
const double REGULARIZATION_L2 = 1e-5;     // L2 penalty strength
const double VALIDATION_SPLIT  = 0.2;      // Percentage of data used for validation

// Activation function selection
enum class ActType { SIGMOID, TANH };
const ActType ACTIVATION = ActType::SIGMOID;

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

// Generic activation depending on ACTIVATION
double activate(double x) {
    switch(ACTIVATION) {
        case ActType::TANH:
            return tanh(x);
        case ActType::SIGMOID:
        default:
            return 1.0 / (1.0 + exp(-x));
    }
}

// Derivative computed from activation output
double activationDerivative(double y) {
    return (ACTIVATION == ActType::SIGMOID) ? y * (1.0 - y) : 1.0 - y * y;
}

// Forward pass returning activations of hidden and output layers
vector<vector<double>> forward(const Network& net, const vector<double>& input) {
    vector<vector<double>> activations(2);

    activations[0].resize(HIDDEN_SIZE);
    for(int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = net.biases[0][i];
        for(int j = 0; j < INPUT_SIZE; j++) {
            sum += net.weights[0][i][j] * input[j];
        }
        activations[0][i] = activate(sum);
    }

    activations[1].resize(OUTPUT_SIZE);
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = net.biases[1][i];
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net.weights[1][i][j] * activations[0][j];
        }
        activations[1][i] = activate(sum);
    }

    return activations;
}

// Standard backpropagation using mean squared error
void backpropagate(Network& net, const vector<double>& input, const vector<double>& target,
                  const vector<vector<double>>& activations, double learning_rate) {
    vector<double> output_deltas(OUTPUT_SIZE);
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        double out = activations[1][i];
        output_deltas[i] = (out - target[i]) * activationDerivative(out);
    }

    vector<double> hidden_deltas(HIDDEN_SIZE);
    for(int i = 0; i < HIDDEN_SIZE; i++) {
        double error = 0.0;
        for(int j = 0; j < OUTPUT_SIZE; j++) {
            error += output_deltas[j] * net.weights[1][j][i];
        }
        double h = activations[0][i];
        hidden_deltas[i] = error * activationDerivative(h);
    }

    for(int i = 0; i < OUTPUT_SIZE; i++) {
        for(int j = 0; j < HIDDEN_SIZE; j++) {
            double grad = output_deltas[i] * activations[0][j] + REGULARIZATION_L2 * net.weights[1][i][j];
            net.momentum_weights[1][i][j] = MOMENTUM * net.momentum_weights[1][i][j] + learning_rate * grad;
            net.weights[1][i][j] -= net.momentum_weights[1][i][j];
        }
        net.momentum_biases[1][i] = MOMENTUM * net.momentum_biases[1][i] + learning_rate * output_deltas[i];
        net.biases[1][i] -= net.momentum_biases[1][i];
    }

    for(int i = 0; i < HIDDEN_SIZE; i++) {
        for(int j = 0; j < INPUT_SIZE; j++) {
            double grad = hidden_deltas[i] * input[j] + REGULARIZATION_L2 * net.weights[0][i][j];
            net.momentum_weights[0][i][j] = MOMENTUM * net.momentum_weights[0][i][j] + learning_rate * grad;
            net.weights[0][i][j] -= net.momentum_weights[0][i][j];
        }
        net.momentum_biases[0][i] = MOMENTUM * net.momentum_biases[0][i] + learning_rate * hidden_deltas[i];
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

            step_line >> step_word >> action >> colon;

            int mapped_action = (action == 3) ? 2 : action;
            if (mapped_action >= 0 && mapped_action <= 2) {
                vector<double> targets(OUTPUT_SIZE, 0.0);
                targets[mapped_action] = 1.0;

                unordered_map<int, double> ram_values;
                string ram_entry;
                while (step_line >> ram_entry) {
                    if (ram_entry[0] == 'R') {
                        size_t u = ram_entry.find('_');
                        if (u != string::npos) {
                            try {
                                int ram_pos = stoi(ram_entry.substr(1, u - 1));
                                double ram_val = stod(ram_entry.substr(u + 1));
                                ram_values[ram_pos] = ram_val;
                            } catch (...) {
                                continue;
                            }
                        }
                    }
                }

                vector<double> inputs(INPUT_SIZE, 1.0);
                for(size_t idx = 0; idx < RAM_POSITIONS.size(); ++idx) {
                    int pos = RAM_POSITIONS[idx];
                    if (ram_values.count(pos)) {
                        inputs[idx] = ram_values[pos];
                    }
                }

                data.push_back({inputs, targets});
                action_mapping[mapped_action]++;
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
        for(int input = 0; input < INPUT_SIZE; input++) {
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
    vector<TrainingData> all_data = loadTrainingData("../agents/datos.csv");
    cout << "Loaded " << all_data.size() << " training samples" << endl;

    if(all_data.empty()) {
        cerr << "No training data found!" << endl;
        return 1;
    }
    
    // Print initial distribution of actions
    vector<int> initial_actions(3, 0);
    for(const auto& sample : all_data) {
        int action = max_element(sample.targets.begin(), sample.targets.end()) - sample.targets.begin();
        initial_actions[action]++;
    }
    cout << "Initial action distribution: [";
    for(int i = 0; i < 3; i++) {
        cout << (initial_actions[i] * 100.0 / all_data.size()) << "% ";
    }
    cout << "]" << endl;

    random_device rd;
    mt19937 gen(rd());

    // Split into training and validation sets
    shuffle(all_data.begin(), all_data.end(), gen);
    size_t val_count = static_cast<size_t>(all_data.size() * VALIDATION_SPLIT);
    vector<TrainingData> validation_data(all_data.begin(), all_data.begin() + val_count);
    vector<TrainingData> training_data(all_data.begin() + val_count, all_data.end());
    cout << "Training samples: " << training_data.size()
         << " - Validation samples: " << validation_data.size() << endl;
    
    double best_val_accuracy = 0.0;
    int epochs_without_improvement = 0;
    double learning_rate = LEARNING_RATE;
    
    for(int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        shuffle(training_data.begin(), training_data.end(), gen);
        
        double total_error = 0.0;
        int correct_predictions = 0;
        
        for(size_t i = 0; i < training_data.size(); i += BATCH_SIZE) {
            size_t batch_end = min(i + BATCH_SIZE, training_data.size());
            
            for(size_t j = i; j < batch_end; j++) {
                vector<vector<double>> activations = forward(net, training_data[j].inputs);
                
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

        // Validation metrics
        double val_error = 0.0;
        int val_correct = 0;
        for(const auto& sample : validation_data) {
            vector<vector<double>> pred = forward(net, sample.inputs);
            for(int k = 0; k < OUTPUT_SIZE; k++) {
                val_error += pow(sample.targets[k] - pred[1][k], 2);
            }
            int p = max_element(pred[1].begin(), pred[1].end()) - pred[1].begin();
            int a = max_element(sample.targets.begin(), sample.targets.end()) - sample.targets.begin();
            if(p == a) val_correct++;
        }
        double val_accuracy = static_cast<double>(val_correct) / validation_data.size();
        double val_avg_error = val_error / validation_data.size();

        if(epoch % 10 == 0) {
            cout << "Epoch " << epoch
                 << " - TrainErr: " << avg_error
                 << " - TrainAcc: " << (accuracy * 100) << "%"
                 << " - ValErr: " << val_avg_error
                 << " - ValAcc: " << (val_accuracy * 100) << "%"
                 << " - LR: " << learning_rate << endl;
        }
        
        if(val_accuracy > best_val_accuracy) {
            best_val_accuracy = val_accuracy;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement++;
            if(epochs_without_improvement % 10 == 0 && learning_rate > MIN_LEARNING_RATE) {
                learning_rate *= 0.5;
                cout << "Reducing learning rate to " << learning_rate << endl;
            }
            if(epochs_without_improvement >= 30) {
                cout << "\nEarly stopping triggered after " << epoch << " epochs" << endl;
                cout << "Best validation accuracy: " << (best_val_accuracy * 100) << "%" << endl;
                break;
            }
        }
    }
    
    cout << "\nTrained weights for game agent:" << endl;
    printWeights(net);
    
    return 0;
}