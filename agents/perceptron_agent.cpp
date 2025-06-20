#include <iostream>
#include <cmath>
#include <cstdint>
#include <SDL/SDL.h>
#include "../src/ale_interface.hpp"

// Constants
constexpr uint32_t maxSteps = 7500;

///////////////////////////////////////////////////////////////////////////////
/// IA's
///////////////////////////////////////////////////////////////////////////////
const unsigned s_kinputs = 26; // Updated for 25 inputs + bias
const unsigned s_numIAs = 1;   // We only need one IA
double s_w[s_numIAs][3][s_kinputs] = {
   { { -32784.86 , -36.58 , -995.66 , -4.99 , -11.25 , -13.84 , 25.27 , -19.53 , -38.45 , -24.02 , -55.15 , 16.66 , -17.22 , 6603.63 , -57.88 , 41.69 , 537.40 , -35.85 , -8.40 , 135.81 , -109.19 , -9.11 , 94.08 , 75.02 , -17.42 , -15.14} 
   , { -36011.70 , -51.98 , -1714.19 , -22.59 , -15.54 , -38.56 , -22.26 , -30.00 , -46.98 , -54.17 , -2.17 , 16.54 , -29.79 , 6919.31 , -23.89 , 58.23 , 577.96 , -53.53 , -6.36 , -11.70 , -104.75 , -20.41 , 150.38 , 171.59 , -14.67 , -22.41} 
   , { -713.78 , -82.61 , 164.31 , 69.37 , 256.22 , 127.42 , -28.69 , 256.75 , -136.55 , 367.51 , -444.34 , -132.91 , 309.45 , -4.38 , -196.82 , -56.80 , -990.89 , -17.39 , 366.07 , -735.43 , 428.23 , 366.72 , 253.30 , 254.82 , 144.00 , 68.68} 
   }
};

///////////////////////////////////////////////////////////////////////////////
// Funcion Hipótesis (H)                               
///////////////////////////////////////////////////////////////////////////////
unsigned h(unsigned wn, unsigned dir, const double* st) {
   double hval = 0;
   for(unsigned i=1; i < s_kinputs; i++) {
      hval += st[i-1] * s_w[wn][dir][i];
   }
   return (hval + s_w[wn][dir][0] > 0) ? 1 : 0;
}

///////////////////////////////////////////////////////////////////////////////
// Get RAM Values                            
///////////////////////////////////////////////////////////////////////////////
double* getRamInputs(const std::vector<int>& ram_positions, const ALERAM& ram) {
   size_t size = ram_positions.size();
   double* ram_values = new double[size];

   for (size_t i = 0; i < size; ++i) {
      int pos = ram_positions[i];
      ram_values[i] = (pos >= 0 && pos < ram.size()) ? static_cast<double>(ram.get(pos)) : 1.0;
   }

   return ram_values;
}

int l, r, s, n = l = r = s = 0;
///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step (Automatic)
///////////////////////////////////////////////////////////////////////////////
reward_t automaticAgentStep(ALEInterface& alei, const int& ia, const std::vector<int>& ram_positions) {
   reward_t reward{0}; 
   unsigned bleft, bright, bshoot;
   const auto& ram = alei.getRAM();

   double* inputs = getRamInputs(ram_positions, ram);

   bleft = h(ia, 0, inputs);
   bright = h(ia, 1, inputs);
   bshoot = h(ia, 2, inputs);

   delete[] inputs;  // Clean up allocated memory

   if      (bleft && !bright && !bshoot) { reward = alei.act(PLAYER_A_LEFT);  l++;  }
   else if (!bleft && bright && !bshoot) { reward = alei.act(PLAYER_A_RIGHT); r++;  }
   else if (!bleft && !bright && bshoot) { reward = alei.act(PLAYER_A_FIRE);  s++;  }
   else                                  { reward = alei.act(PLAYER_A_NOOP);  n++;  }
   
   return reward;
}

///////////////////////////////////////////////////////////////////////////////
/// Automatic
///////////////////////////////////////////////////////////////////////////////
void automaticMode(ALEInterface& alei, reward_t& totalReward, const int& ia) {
   // Convert hex positions to decimal
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

   std::srand(static_cast<uint32_t>(std::time(0)));

   automaticMode(alei, totalReward, 0);  // Using IA index 0

   std::cout << "Action counts - Left: " << l << " Right: " << r 
            << " Shoot: " << s << " Noop: " << n << std::endl;
   
   return 0;
}