#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_scancode.h>
#include <map>
#include <memory>

class KeyboardHandler
{
  public:
    KeyboardHandler() = default;

    bool getKeyState(SDL_Scancode scancode);
    std::pair<int, int> getMouseDelta();
    int handleInput(SDL_Event event);

  private:
    std::map<SDL_Scancode, bool> _key_states;
    int _mouse_dx;
    int _mouse_dy;
};