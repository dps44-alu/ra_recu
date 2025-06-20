#include <iostream>
#include <cmath>
#include <cstdint>
#include <SDL/SDL.h>
#include <fstream>
#include "../src/ale_interface.hpp"

// Constants
constexpr uint32_t maxSteps = 7500;

///////////////////////////////////////////////////////////////////////////////
/// Neural Network Constants and Weights
///////////////////////////////////////////////////////////////////////////////
const unsigned s_kinputs = 26;    // Input layer size
const unsigned s_hidden = 30;     // Hidden layer size
const unsigned s_output = 3;      // Output layer size (left, right, shoot)
const unsigned s_numIAs = 1;      // We only need one IA

// Weights and biases will be loaded from file
double s_hidden_weights[s_numIAs][s_kinputs][s_hidden];
double s_output_weights[s_numIAs][s_hidden][s_output];
double s_hidden_bias[s_numIAs][s_hidden];
double s_output_bias[s_numIAs][s_output];

///////////////////////////////////////////////////////////////////////////////
/// Load Neural Network Weights
///////////////////////////////////////////////////////////////////////////////
bool loadWeights(const std::string& filename, int ia) {
   std::ifstream file(filename);
   if (!file.is_open()) {
      std::cerr << "Could not open weights file: " << filename << std::endl;
      return false;
   }

   // Read and verify network architecture
   unsigned in_size, hid_size, out_size;
   file >> in_size >> hid_size >> out_size;
   if (in_size != s_kinputs || hid_size != s_hidden || out_size != s_output) {
      std::cerr << "Network architecture mismatch in weights file" << std::endl;
      return false;
   }

   // Load hidden weights
   for(unsigned i = 0; i < s_kinputs; ++i) {
      for(unsigned j = 0; j < s_hidden; ++j) {
         file >> s_hidden_weights[ia][i][j];
      }
   }

   // Load output weights
   for(unsigned i = 0; i < s_hidden; ++i) {
      for(unsigned j = 0; j < s_output; ++j) {
         file >> s_output_weights[ia][i][j];
      }
   }

   // Load biases
   for(unsigned i = 0; i < s_hidden; ++i) {
      file >> s_hidden_bias[ia][i];
   }
   for(unsigned i = 0; i < s_output; ++i) {
      file >> s_output_bias[ia][i];
   }

   file.close();
   return true;
}

///////////////////////////////////////////////////////////////////////////////
/// Sigmoid Activation Function
///////////////////////////////////////////////////////////////////////////////
double sigmoid(double x) {
   return 1.0 / (1.0 + std::exp(-x));
}

///////////////////////////////////////////////////////////////////////////////
/// Neural Network Forward Pass
///////////////////////////////////////////////////////////////////////////////
std::vector<double> forwardPass(const std::vector<double>& inputs, const int& ia) {
   // Hidden layer
   std::vector<double> hidden_activations(s_hidden);
   for(unsigned j = 0; j < s_hidden; ++j) {
      double sum = s_hidden_bias[ia][j];
      for(unsigned i = 0; i < s_kinputs; ++i) {
         sum += inputs[i] * s_hidden_weights[ia][i][j];
      }
      hidden_activations[j] = sigmoid(sum);
   }

   // Output layer
   std::vector<double> outputs(s_output);
   for(unsigned k = 0; k < s_output; ++k) {
      double sum = s_output_bias[ia][k];
      for(unsigned j = 0; j < s_hidden; ++j) {
         sum += hidden_activations[j] * s_output_weights[ia][j][k];
      }
      outputs[k] = sigmoid(sum);
   }

   return outputs;
}

///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step (Automatic)
///////////////////////////////////////////////////////////////////////////////
reward_t automaticAgentStep(ALEInterface& alei, const int& ia, const std::vector<int>& ram_positions) {
   reward_t reward{0}; 
   const auto& ram = alei.getRAM();

   // Prepare inputs
   std::vector<double> inputs(s_kinputs);
   for(size_t i = 0; i < ram_positions.size(); ++i) {
      int pos = ram_positions[i];
      inputs[i] = (pos >= 0 && pos < ram.size()) ? static_cast<double>(ram.get(pos)) : 1.0;
   }

   // Get network outputs
   std::vector<double> outputs = forwardPass(inputs, ia);

   // Find action with highest activation
   int action = std::max_element(outputs.begin(), outputs.end()) - outputs.begin();

   // Convert network output to game action
   switch(action) {
      case 0: // Left
         reward = alei.act(PLAYER_A_LEFT);
         break;
      case 1: // Right
         reward = alei.act(PLAYER_A_RIGHT);
         break;
      case 2: // Shoot
         reward = alei.act(PLAYER_A_FIRE);
         break;
      default:
         reward = alei.act(PLAYER_A_NOOP);
   }

   return reward;
}

///////////////////////////////////////////////////////////////////////////////
/// Automatic Mode
///////////////////////////////////////////////////////////////////////////////
void automaticMode(ALEInterface& alei, reward_t& totalReward, const int& ia) {
   // Convert hex positions to decimal (same as before)
   std::vector<int> ram_positions = {
      0x10, 0x30, 0x40, 0x21, 0x31, 0x22, 0x32, 0x42, 0x13, 0x44,
      0x25, 0x45, 0x55, 0x16, 0x26, 0x46, 0x07, 0x47, 0x08, 0x09,
      0x49, 0x4a, 0x4b, 0x0f, 0x1f, 0x2f
   };

   uint32_t step{};
   while (!alei.game_over() && step < maxSteps) { 
      totalReward += automaticAgentStep(alei, ia, ram_positions);
      ++step;
   }

   std::cout << "Steps: " << step << "\nReward: " << totalReward << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
/// Print usage and exit
///////////////////////////////////////////////////////////////////////////////
void usage(char const* pname) {
   std::cerr << "\nUSAGE:\n" << "   " << pname << " <romfile>\n";
   exit(-1);
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   if (argc != 2) usage(argv[0]);

   ALEInterface alei{};
   reward_t totalReward{};

   alei.setInt("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool("display_screen", true);
   alei.setBool("sound", true);
   alei.loadROM(argv[1]);

   // Load neural network weights
   if (!loadWeights("../training/best_weights.txt", 0)) {
      std::cerr << "Failed to load weights" << std::endl;
      return 1;
   }

   automaticMode(alei, totalReward, 0);  // Using IA index 0
   
   return 0;
}