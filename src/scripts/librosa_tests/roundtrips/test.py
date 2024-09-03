import keyboard

def on_key_event(e):
    print(f'Key {e.name} {e.event_type}')

keyboard.hook(on_key_event)

keyboard.wait('esc')  # This will wait for the 'esc' key to be pressed and then exit the program
