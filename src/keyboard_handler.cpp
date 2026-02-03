#include "keyboard_handler.h"

bool KeyboardHandler::getKeyState(SDL_Scancode scancode)
{
    return _key_states[scancode];
}

std::pair<int, int> KeyboardHandler::getMouseDelta()
{
    std::pair<int, int> result{_mouse_dx, _mouse_dy};
    _mouse_dx = 0;
    _mouse_dy = 0;
    return result;
}

int KeyboardHandler::handleInput(SDL_Event event)
{

    switch (event.type)
    {
    case SDL_EVENT_KEY_DOWN:
        // std::cerr << "Handle keydown\n";
        _key_states[event.key.scancode] = true;
        break;
    case SDL_EVENT_KEY_UP:
        // std::cerr << "Handle keyup\n";
        _key_states[event.key.scancode] = false;
        break;
    case SDL_EVENT_MOUSE_MOTION:
        // std::cerr << mouse_dx << std::endl;
        _mouse_dx = event.motion.xrel;
        _mouse_dy = event.motion.yrel;
        break;
    }

    return 0;
}