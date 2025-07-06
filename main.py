import keyboard
import pygame
import threading
import json
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sounddevice as sd
import numpy as np
import soundfile as sf
import webbrowser
import ctypes

CONFIG_FILE = "config.json"
SOUND_FOLDER = "sounds"

devices = sd.query_devices()
pyaudio_output_devices = [d for d in devices if d['max_output_channels'] > 0]

is_playing = False
current_sound = None
selected_output_index = None
selected_output2_index = None

config = {}  # глобальна змінна для збереження конфігу

play_threads = []
stop_event = threading.Event()
def toggle_sound(path):
    global is_playing, current_sound, play_threads, active_streams, stop_event
    if 'active_streams' not in globals():
        active_streams = []
    if is_playing and current_sound == path:
        stop_event.set()  # Сигнал усім потокам зупинитися
        with active_streams_lock:
            for stream in active_streams[:]:
                try:
                    if hasattr(stream, 'active') and stream.active:
                        try:
                            stream.stop()
                        except Exception as e:
                            print(f"[!] Error stopping stream: {e}")
                    if hasattr(stream, 'closed') and not stream.closed:
                        try:
                            stream.close()
                        except Exception as e:
                            print(f"[!] Error closing stream: {e}")
                except Exception as e:
                    print(f"[!] General error with stream: {e}")
                try:
                    active_streams.remove(stream)
                except Exception:
                    pass
            active_streams.clear()
        is_playing = False
        stop_event.clear()
        return

    def _play():
        global is_playing, current_sound, stop_event
        stop_event.clear()
        if os.path.exists(path):
            try:
                sound_data, fs = sf.read(path, dtype='float32')
                if len(sound_data.shape) == 1:
                    sound_data = np.expand_dims(sound_data, axis=1)

                threads = []
                print(f"[DEBUG] selected_output_index: {selected_output_index}, selected_output2_index: {selected_output2_index}")
                if selected_output_index is not None:
                    t1 = threading.Thread(target=play_on_device, args=(sound_data, fs, selected_output_index), daemon=True)
                    threads.append(t1)
                    t1.start()
                if selected_output2_index is not None and selected_output2_index != selected_output_index:
                    t2 = threading.Thread(target=play_on_device, args=(sound_data, fs, selected_output2_index), daemon=True)
                    threads.append(t2)
                    t2.start()

                play_threads = threads
                current_sound = path
                is_playing = True
                for t in threads:
                    t.join()
                with active_streams_lock:
                    for stream in active_streams[:]:
                        try:
                            if hasattr(stream, 'active') and stream.active:
                                try:
                                    stream.stop()
                                except Exception as e:
                                    print(f"[!] Error stopping stream: {e}")
                            if hasattr(stream, 'closed') and not stream.closed:
                                try:
                                    stream.close()
                                except Exception as e:
                                    print(f"[!] Error closing stream: {e}")
                        except Exception as e:
                            print(f"[!] General error with stream: {e}")
                        try:
                            active_streams.remove(stream)
                        except Exception:
                            pass
                    active_streams.clear()
                is_playing = False
            except Exception as e:
                print(f"[!] Playback error: {e}")
        else:
            print(f"[!] File not found: {path}")

    threading.Thread(target=_play, daemon=True).start()

active_streams = []
active_streams_lock = threading.Lock()

def load_config():
    global config, selected_output_index, selected_output2_index
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    config = {}
                else:
                    config = json.loads(content)
                    if '__device__' in config:
                        selected_output_index = config['__device__'].get('output1')
                        selected_output2_index = config['__device__'].get('output2')
        except json.JSONDecodeError:
            print("[!] Configuration file damaged. Now using blank config.")
            config = {}
    else:
        config = {}

def save_config(cfg):
    cfg['__device__'] = {
        'output1': selected_output_index,
        'output2': selected_output2_index
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)

def play_on_device(sound_data, fs, device_index):
    global active_streams, stop_event
    try:
        device_info = sd.query_devices(device_index)
        print(f"[PLAY] Playing on: {device_info['name']} (index {device_index})")
        target_fs = int(device_info['default_samplerate'])

        if fs != target_fs:
            print(f"[!] Resampling from {fs} Hz to {target_fs} Hz for device {device_index}")
            duration = sound_data.shape[0] / fs
            num_samples = int(duration * target_fs)
            sound_data = np.array([
                np.interp(
                    np.linspace(0, len(channel), num_samples),
                    np.arange(len(channel)),
                    channel
                )
                for channel in sound_data.T
            ], dtype=np.float32).T
        else:
            sound_data = sound_data.astype(np.float32)

        if not sound_data.flags['C_CONTIGUOUS']:
            sound_data = np.ascontiguousarray(sound_data)

        idx = [0]
        def callback(outdata, frames, time, status):
            if stop_event.is_set():
                raise sd.CallbackStop()
            chunk = sound_data[idx[0]:idx[0]+frames]
            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = 0
                raise sd.CallbackStop()
            else:
                outdata[:] = chunk
            idx[0] += frames

        with sd.OutputStream(device=device_index, samplerate=target_fs, channels=sound_data.shape[1], callback=callback) as stream:
            with active_streams_lock:
                active_streams.append(stream)
            stream.start()
            while stream.active:
                if stop_event.is_set():
                    break
                time.sleep(0.05)
    except Exception as e:
        print(f"[!] Error in play_on_device: {e}")

def toggle_sound(path):
    global is_playing, current_sound, play_threads, active_streams, stop_event
    if 'active_streams' not in globals():
        active_streams = []
    if is_playing and current_sound == path:
        stop_event.set()  # Сигнал усім потокам зупинитися
        with active_streams_lock:
            for stream in active_streams[:]:
                try:
                    if hasattr(stream, 'active') and stream.active:
                        try:
                            stream.stop()
                        except Exception as e:
                            print(f"[!] Error stopping stream: {e}")
                    if hasattr(stream, 'closed') and not stream.closed:
                        try:
                            stream.close()
                        except Exception as e:
                            print(f"[!] Error closing stream: {e}")
                except Exception as e:
                    print(f"[!] General error with stream: {e}")
                try:
                    active_streams.remove(stream)
                except Exception:
                    pass
            active_streams.clear()
        is_playing = False
        stop_event.clear()
        return

    def _play():
        global is_playing, current_sound, stop_event
        stop_event.clear()
        if os.path.exists(path):
            try:
                sound_data, fs = sf.read(path, dtype='float32')
                if len(sound_data.shape) == 1:
                    sound_data = np.expand_dims(sound_data, axis=1)

                threads = []
                print(f"[DEBUG] selected_output_index: {selected_output_index}, selected_output2_index: {selected_output2_index}")
                if selected_output_index is not None:
                    t1 = threading.Thread(target=play_on_device, args=(sound_data, fs, selected_output_index), daemon=True)
                    threads.append(t1)
                    t1.start()
                if selected_output2_index is not None and selected_output2_index != selected_output_index:
                    t2 = threading.Thread(target=play_on_device, args=(sound_data, fs, selected_output2_index), daemon=True)
                    threads.append(t2)
                    t2.start()

                play_threads = threads
                current_sound = path
                is_playing = True
                for t in threads:
                    t.join()
                with active_streams_lock:
                    for stream in active_streams[:]:
                        try:
                            if hasattr(stream, 'active') and stream.active:
                                try:
                                    stream.stop()
                                except Exception as e:
                                    print(f"[!] Error stopping stream: {e}")
                            if hasattr(stream, 'closed') and not stream.closed:
                                try:
                                    stream.close()
                                except Exception as e:
                                    print(f"[!] Error closing stream: {e}")
                        except Exception as e:
                            print(f"[!] General error with stream: {e}")
                        try:
                            active_streams.remove(stream)
                        except Exception:
                            pass
                    active_streams.clear()
                is_playing = False
            except Exception as e:
                print(f"[!] Playback error: {e}")
        else:
            print(f"[!] File not found: {path}")

    threading.Thread(target=_play, daemon=True).start()

def bind_keys():
    for hotkey, sound_path in config.items():
        if hotkey == '__device__':
            continue
        keyboard.add_hotkey(hotkey, lambda p=sound_path: toggle_sound(p))
        print(f"[✓] Linked: {hotkey} → {sound_path}")

def gui_add_keybind():
    key = None
    top = tk.Toplevel(root)
    top.title("Press a Key")
    top.geometry("300x120")

    label = tk.Label(top, text="Press a key...", font=("Consolas", 12))
    label.pack(pady=10)

    key_display = tk.Label(top, text="", font=("Consolas", 14), fg="green")
    key_display.pack(pady=5)

    def on_key(event):
        nonlocal key
        try:
            key = event.keysym.lower()
            keyboard.parse_hotkey(key)
            key_display.config(text=f"Selected: {key}")
            top.after(500, top.destroy)
        except ValueError:
            messagebox.showerror("Error", f"Key '{event.keysym}' is not supported.")
            top.destroy()
            return

    top.bind("<Key>", on_key)
    top.grab_set()
    root.wait_window(top)

    if not key:
        return

    if key in config:
        messagebox.showwarning("Warning", f"Key {key} is already assigned.")
        return

    path = filedialog.askopenfilename(initialdir=SOUND_FOLDER, title="Select Sound File",
                                      filetypes=(("Audio Files", "*.mp3 *.wav *.flac *.ogg"),))
    if not path:
        return

    config[key] = path
    save_config(config)
    keyboard.add_hotkey(key, lambda p=path: toggle_sound(p))
    messagebox.showinfo("Added", f"Linked: {key} to {os.path.basename(path)}")
    update_listbox()

def update_listbox():
    listbox.delete(0, tk.END)
    for key, path in config.items():
        if key == '__device__':
            continue
        listbox.insert(tk.END, f"{key} → {os.path.basename(path)}")

def gui_remove_keybind():
    selected = listbox.curselection()
    if not selected:
        return
    item = listbox.get(selected[0])
    key = item.split(" → ")[0]
    if key in config:
        keyboard.remove_hotkey(key)
        del config[key]
        save_config(config)
        update_listbox()
        messagebox.showinfo("Deleted", f"Deleted: {key}")

def populate_audio_devices():
    output_names = ["None"] + [d['name'] for d in pyaudio_output_devices]
    # Обидва комбобокси використовують тільки вихідні пристрої
    output_combo['values'] = output_names
    output2_combo['values'] = output_names

    # Set selection based on saved index, or default to "None"
    if selected_output_index is not None:
        for i, d in enumerate(pyaudio_output_devices):
            if d['index'] == selected_output_index:
                output_combo.current(i + 1)
                break
        else:
            output_combo.current(0)
    else:
        output_combo.current(0)

    if selected_output2_index is not None:
        for i, d in enumerate(pyaudio_output_devices):
            if d['index'] == selected_output2_index:
                output2_combo.current(i + 1)
                break
        else:
            output2_combo.current(0)
    else:
        output2_combo.current(0)

def update_selected_devices(event=None):
    global selected_output_index, selected_output2_index
    if output_combo.current() == 0:
        selected_output_index = None
    else:
        selected_output_index = pyaudio_output_devices[output_combo.current() - 1]['index']
    if output2_combo.current() == 0:
        selected_output2_index = None
    else:
        selected_output2_index = pyaudio_output_devices[output2_combo.current() - 1]['index']
    print(f"Speaker: {output_combo.get()} ({selected_output_index})")
    print(f"Virtual Microphone: {output2_combo.get()} ({selected_output2_index})")
    save_config(config)

def open_tutorial():
    webbrowser.open("https://pysoundboardtutorial.carrd.co/")

if __name__ == "__main__":
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

    if not os.path.exists(SOUND_FOLDER):
        os.makedirs(SOUND_FOLDER)

    load_config()

    root = tk.Tk()
    root.title("PySoundboard")
    root.geometry("500x550")

    listbox = tk.Listbox(root, font=("Consolas", 12))
    listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    frame = tk.Frame(root)
    frame.pack(pady=5)

    btn_add = tk.Button(frame, text="Add", command=gui_add_keybind, width=15)
    btn_add.pack(side=tk.LEFT, padx=5)

    btn_remove = tk.Button(frame, text="Delete", command=gui_remove_keybind, width=15)
    btn_remove.pack(side=tk.LEFT, padx=5)

    tutorial_btn = tk.Button(root, text="📘 Open Tutorial", command=open_tutorial, fg="blue")
    tutorial_btn.pack(pady=5)

    audio_frame = tk.LabelFrame(root, text="Audio Devices")
    audio_frame.pack(fill=tk.X, padx=10, pady=10)

    tk.Label(audio_frame, text="🔊 Virtual Microphone:").pack(side=tk.LEFT, padx=5)
    output_combo = ttk.Combobox(audio_frame, width=25)
    output_combo.pack(side=tk.LEFT, padx=5)

    tk.Label(audio_frame, text="🎧 Speaker:").pack(side=tk.LEFT, padx=5)
    output2_combo = ttk.Combobox(audio_frame, width=25)
    output2_combo.pack(side=tk.LEFT, padx=5)

    output_combo.bind("<<ComboboxSelected>>", update_selected_devices)
    output2_combo.bind("<<ComboboxSelected>>", update_selected_devices)

    populate_audio_devices()
    update_selected_devices()
    bind_keys()
    update_listbox()

    tk.Label(root, text="Esc — close program", fg="gray").pack(pady=5)

    keyboard.add_hotkey("esc", lambda: root.destroy())
    root.mainloop()
