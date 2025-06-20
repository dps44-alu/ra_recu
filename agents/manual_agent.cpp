#include <iostream>
#include <cmath>
#include <cstdint>
#include <SDL/SDL.h>
#include "../src/ale_interface.hpp"

// Constants
constexpr uint32_t maxSteps = 7500;

///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step (Manual)
///////////////////////////////////////////////////////////////////////////////
reward_t manualAgentStep(ALEInterface& alei, int& action) {
   reward_t reward{0}; 
   Uint8* keyState = SDL_GetKeyState(NULL);

   if      (keyState[SDLK_LEFT])  { reward = alei.act(PLAYER_A_LEFT);  action = 1;  }
   else if (keyState[SDLK_RIGHT]) { reward = alei.act(PLAYER_A_RIGHT); action = 2;  }
   else if (keyState[SDLK_SPACE]) { reward = alei.act(PLAYER_A_FIRE);  action = 3;  }
   else                           { reward = alei.act(PLAYER_A_NOOP);   action = 0; }

   return reward;
}

///////////////////////////////////////////////////////////////////////////////
/// Manual
///////////////////////////////////////////////////////////////////////////////
void manualMode(ALEInterface& alei, reward_t& totalReward) {
   std::ofstream logFile("datos.csv", std::ios::app);
   if (!logFile.is_open()) {
      std::cerr << "Failed to open log file." << std::endl;
      return;
   }

   const auto& ram = alei.getRAM();
   std::vector<uint8_t> prevRAM(ram.size());
   for (size_t i = 0; i < ram.size(); ++i) {
      prevRAM[i] = ram.get(i);
   }

   uint32_t step{};
   int action = 0;
   while (!alei.game_over() && step < maxSteps) {
      totalReward += manualAgentStep(alei, action);
      ++step;

      logFile << "Step " << action << " :";
      bool hasChanges = false;
      for (size_t i = 0; i < ram.size(); ++i) {
         uint8_t currentValue = ram.get(i);
         if (currentValue != prevRAM[i]) {
               logFile << " R" << i << "_" << int(currentValue);
               prevRAM[i] = currentValue;
               hasChanges = true;
         }
      }
      if (hasChanges) logFile << "\n";
   }

   std::cout << "Steps: " << step << "\nReward: " << totalReward << std::endl;
   logFile.close();
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

   manualMode(alei, totalReward);
   
   return 0;
}