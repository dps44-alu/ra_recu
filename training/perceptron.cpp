#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <ctime>
#include <random>
#include <numeric>
#include <iomanip>

using namespace std;

// Training parameters
const unsigned s_kinputs = 26;
const int MAX_ITERATIONS = 500000;
const double LEARNING_RATE = 0.001;
const double CONVERGENCE_THRESHOLD = 0.98;
const int MAX_IMPRO = 50000;
const int PROGRESS_UPDATE_FREQ = 1000;

// Important RAM positions
const int Y_POS_1 = 0x10;
const int Y_POS_2 = 0x16;
const int SHOOT_POS = 0x4F;
// Add after the other RAM positions
const int BULLET_POS_1 = 0x25;
const int BULLET_POS_2 = 0x26;
const double BULLET_WEIGHT_MULTIPLIER = 15.0;  // Higher weight for bullets
const double POSITION_WEIGHT_MULTIPLIER = 5.0;
const double SHOOT_WEIGHT_MULTIPLIER = 10.0;

// Progress bar settings
const int BAR_WIDTH = 50;

// Training data structure
struct TrainingData {
    vector<double> inputs;
    vector<double> position_weights;
    int label_left;
    int label_right;
    int label_shoot;
};

// Print weights in the correct format
void printWeights(const vector<double>& W_left, const vector<double>& W_right, const vector<double>& W_shoot) {
    cout << "{ { ";
    for (size_t i = 0; i < W_left.size(); ++i) {
        cout << W_left[i];
        if (i < W_left.size() - 1) cout << " , ";
    }
    cout << "} \n, { ";
    for (size_t i = 0; i < W_right.size(); ++i) {
        cout << W_right[i];
        if (i < W_right.size() - 1) cout << " , ";
    }
    cout << "} \n, { ";
    for (size_t i = 0; i < W_shoot.size(); ++i) {
        cout << W_shoot[i];
        if (i < W_shoot.size() - 1) cout << " , ";
    }
    cout << "} \n}" << endl;
}

int hexToDecimal(const string& hex) {
    return stoi(hex, nullptr, 16);
}

void loadTrainingData(const string& filename, const unordered_set<int>& ram_positions, 
                     vector<TrainingData>& leftData, vector<TrainingData>& rightData,
                     vector<TrainingData>& shootData, vector<TrainingData>& nothingData) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open file: " << filename << endl;
        return;
    }

    unordered_map<int, double> position_weights;
    for (int pos : ram_positions) {
        double weight = 1.0;
        if (pos == Y_POS_1 || pos == Y_POS_2) {
            weight = POSITION_WEIGHT_MULTIPLIER;
            cout << "Higher weight applied to Y position " << hex << pos << dec << endl;
        }
        else if (pos == SHOOT_POS) {
            weight = SHOOT_WEIGHT_MULTIPLIER;
            cout << "Higher weight applied to shoot position " << hex << pos << dec << endl;
        }
        else if (pos == BULLET_POS_1 || pos == BULLET_POS_2) {
            weight = BULLET_WEIGHT_MULTIPLIER;
            cout << "Higher weight applied to bullet position " << hex << pos << dec << endl;
        }
        position_weights[pos] = weight;
    }

    unordered_map<int, double> ram_values;
    string line;
    int line_number = 0;
    int valid_entries = 0;
    int skipped_entries = 0;

    while (getline(file, line)) {
        line_number++;
        try {
            if (line.rfind("Step", 0) == 0) {
                istringstream step_line(line);
                string step_word;
                int action;
                
                // Parse Step and action
                if (!(step_line >> step_word >> action)) {
                    cerr << "Error parsing Step and action on line " << line_number << endl;
                    skipped_entries++;
                    continue;
                }

                // Parse colon
                string colon;
                if (!(step_line >> colon) || colon != ":") {
                    cerr << "Error parsing colon on line " << line_number << endl;
                    skipped_entries++;
                    continue;
                }

                ram_values.clear();
                string ram_entry;
                bool valid_line = true;

                while (step_line >> ram_entry) {
                    if (ram_entry[0] == 'R') {
                        size_t underscore_pos = ram_entry.find('_');
                        if (underscore_pos == string::npos) {
                            cerr << "Invalid RAM entry format at line " << line_number << ": " << ram_entry << endl;
                            valid_line = false;
                            break;
                        }

                        try {
                            int ram_pos = stoi(ram_entry.substr(1, underscore_pos - 1));
                            double ram_val = stod(ram_entry.substr(underscore_pos + 1));
                            ram_values[ram_pos] = ram_val;
                        } catch (const exception& e) {
                            cerr << "Error parsing RAM values at line " << line_number << ": " << ram_entry << endl;
                            valid_line = false;
                            break;
                        }
                    }
                }

                if (!valid_line) {
                    skipped_entries++;
                    continue;
                }

                vector<double> inputs;
                vector<double> weights;
                inputs.push_back(1.0);  // Bias term
                weights.push_back(1.0);

                for (int pos : ram_positions) {
                    if (ram_values.find(pos) != ram_values.end()) {
                        inputs.push_back(ram_values[pos]);
                    } else {
                        inputs.push_back(1.0);
                    }
                    weights.push_back(position_weights[pos]);
                }

                int label_left = (action == 1) ? 1 : -1;
                int label_right = (action == 2) ? 1 : -1;
                int label_shoot = (action == 3) ? 1 : -1;

                TrainingData data = {inputs, weights, label_left, label_right, label_shoot};
                
                if (label_left == 1) {
                    leftData.push_back(data);
                } else if (label_right == 1) {
                    rightData.push_back(data);
                } else if (label_shoot == 1) {
                    shootData.push_back(data);
                } else {
                    nothingData.push_back(data);
                }

                valid_entries++;
            }
        } catch (const exception& e) {
            cerr << "Error processing line " << line_number << ": " << e.what() << endl;
            cerr << "Line content: " << line << endl;
            skipped_entries++;
            continue;
        }
    }

    cout << "\nData loading summary:" << endl;
    cout << "Total lines processed: " << line_number << endl;
    cout << "Valid entries: " << valid_entries << endl;
    cout << "Skipped entries: " << skipped_entries << endl;
    cout << "\nData distribution:" << endl;
    cout << "Left moves: " << leftData.size() << endl;
    cout << "Right moves: " << rightData.size() << endl;
    cout << "Shoot actions: " << shootData.size() << endl;
    cout << "No action: " << nothingData.size() << endl;

    file.close();
}
           

vector<TrainingData> modifyData(vector<TrainingData>& data1, const vector<TrainingData>& data2, double quantity) {
    if (quantity < 0.0 || quantity > 1.0) {
        throw invalid_argument("Quantity must be between 0.0 and 1.0");
    }

    random_device rd;
    mt19937 gen(rd());

    size_t count = static_cast<size_t>(data1.size() * quantity);
    vector<size_t> indices(data2.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);

    for (size_t i = 0; i < count && i < data2.size(); ++i) {
        data1.push_back(data2[indices[i]]);
    }

    return data1;
}

int hypothesis(const vector<double>& weights, const vector<double>& inputs, const vector<double>& position_weights) {
    double hval = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        hval += weights[i] * inputs[i] * position_weights[i];
    }
    return (hval >= 0) ? 1 : -1;
}

void printProgressBar(int iteration, int max_iterations, double accuracy, double avg_accuracy) {
    float progress = static_cast<float>(iteration) / max_iterations;
    int pos = static_cast<int>(BAR_WIDTH * progress);

    cout << "\r[";
    for (int i = 0; i < BAR_WIDTH; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << std::setw(3) << int(progress * 100.0) << "% | "
         << "Iter: " << std::setw(7) << iteration << " | "
         << "Acc: " << std::fixed << std::setprecision(2) << accuracy * 100 << "% | "
         << "Avg: " << avg_accuracy * 100 << "%" << std::flush;
}

void clearLine() {
    cout << "\r" << string(120, ' ') << "\r" << flush;
}

vector<double> trainPerceptron(const vector<TrainingData>& data, int output_index, int max_iterations) {
    cout << "\nInitializing training for " 
         << (output_index == 0 ? "LEFT" : output_index == 1 ? "RIGHT" : "SHOOT") 
         << " movement..." << endl;
    
    random_device rd;
    mt19937 gen(rd());
    
    vector<double> weights(s_kinputs);
    uniform_real_distribution<> dist(-0.05, 0.05);
    for(auto& w : weights) {
        w = dist(gen);
    }
    
    vector<double> best_weights = weights;
    double accuracy = 0.0, best_accuracy = 0.0;
    int iterations_without_improvement = 0;
    
    vector<double> accuracy_history;
    const int HISTORY_WINDOW = 100;
    
    cout << "Starting training with parameters:" << endl;
    cout << "Learning rate: " << LEARNING_RATE << endl;
    cout << "Convergence threshold: " << CONVERGENCE_THRESHOLD << endl;
    cout << "Max iterations without improvement: " << MAX_IMPRO << endl;
    cout << "Training data size: " << data.size() << endl << endl;
    
    time_t start_time = time(nullptr);
    
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        int correct = 0;
        vector<size_t> misclassified;
        
        vector<size_t> indices(data.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), gen);
        
        for (size_t idx : indices) {
            int expected = (output_index == 0) ? data[idx].label_left :
                         (output_index == 1) ? data[idx].label_right :
                                             data[idx].label_shoot;
            
            int predicted = hypothesis(weights, data[idx].inputs, data[idx].position_weights);
            
            if (predicted != expected) {
                misclassified.push_back(idx);
            } else {
                ++correct;
            }
        }
        
        accuracy = static_cast<double>(correct) / data.size();
        accuracy_history.push_back(accuracy);
        if (accuracy_history.size() > HISTORY_WINDOW) {
            accuracy_history.erase(accuracy_history.begin());
        }
        
        double avg_accuracy = 0;
        if (accuracy_history.size() >= HISTORY_WINDOW) {
            avg_accuracy = accumulate(accuracy_history.begin(), accuracy_history.end(), 0.0) / accuracy_history.size();
        }
        
        if (iteration % PROGRESS_UPDATE_FREQ == 0) {
            clearLine();
            printProgressBar(iteration, max_iterations, accuracy, avg_accuracy);
        }
        
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_weights = weights;
            iterations_without_improvement = 0;
            
            if (iteration % (PROGRESS_UPDATE_FREQ * 10) == 0) {
                clearLine();
                cout << "\nNew best accuracy: " << (accuracy * 100) << "% at iteration " << iteration << endl;
            }
        } else {
            iterations_without_improvement++;
        }
        
        if ((avg_accuracy >= CONVERGENCE_THRESHOLD && accuracy_history.size() >= HISTORY_WINDOW) || 
            iterations_without_improvement > MAX_IMPRO) {
            clearLine();
            cout << "\nTraining stopped after " << iteration << " iterations - ";
            if (avg_accuracy >= CONVERGENCE_THRESHOLD) {
                cout << "Convergence threshold reached" << endl;
            } else {
                cout << "No improvement for " << MAX_IMPRO << " iterations" << endl;
            }
            break;
        }
        
        if (!misclassified.empty()) {
            for (size_t idx : misclassified) {
                int expected = (output_index == 0) ? data[idx].label_left :
                             (output_index == 1) ? data[idx].label_right :
                                                 data[idx].label_shoot;
                
                for (size_t i = 0; i < weights.size(); ++i) {
                    weights[i] += LEARNING_RATE * expected * data[idx].inputs[i] * data[idx].position_weights[i];
                }
            }
        }
    }

    time_t end_time = time(nullptr);
    double training_time = difftime(end_time, start_time);
    
    cout << "\nTraining completed:" << endl;
    cout << "Final accuracy: " << (best_accuracy * 100) << "%" << endl;
    cout << "Training time: " << training_time << " seconds" << endl;
    cout << "------------------------------------------" << endl;
    
    return best_weights;
}

int main() {
    vector<string> hex_positions = {
        "10", "30", "40", "21", "31", "22", "32", "42", "13", "44",
        "25", "45", "55", "16", "26", "46", "07", "47", "08", "09",
        "49", "4a", "4b", "0f", "1f", "2f"
    };
    
    unordered_set<int> ram_positions;
    cout << "Using RAM positions (decimal): ";
    for (const string& hex : hex_positions) {
        int decimal = hexToDecimal(hex);
        ram_positions.insert(decimal);
        cout << decimal << " ";
    }
    cout << endl;

    string filename = "../agents/datos.csv";
    vector<TrainingData> leftData, rightData, shootData, nothingData;
    loadTrainingData(filename, ram_positions, leftData, rightData, shootData, nothingData);
    
    if (leftData.empty() || rightData.empty() || shootData.empty() || nothingData.empty()) {
        cerr << "No training data found" << endl;
        return 1;
    }

    cout << "\nTraining perceptron for W_left..." << endl;
    leftData = modifyData(leftData, nothingData, 1.0);
    vector<double> W_left = trainPerceptron(leftData, 0, MAX_ITERATIONS);
    
    cout << "\nTraining perceptron for W_right..." << endl;
    rightData = modifyData(rightData, nothingData, 1.0);
    vector<double> W_right = trainPerceptron(rightData, 1, MAX_ITERATIONS);

    cout << "\nTraining perceptron for W_shoot..." << endl;
    shootData = modifyData(shootData, nothingData, 1.0);
    vector<double> W_shoot = trainPerceptron(shootData, 2, MAX_ITERATIONS);
    
    cout << "\nTrained weights:" << endl;
    printWeights(W_left, W_right, W_shoot);
    
    return 0;
}