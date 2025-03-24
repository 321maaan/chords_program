

from program import playChord, get_user_chords, get_key_from_chords, up_pitch,playChord, key_pitch_change, playChordSequence, get_chord_from_sound

import time
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr  # noise reduction library
from numpy.linalg import norm


        
def main():
   
    """
    Ok so. this is my Program of music.
    It can do alot of stuff, you can input a chord
    play it and change it scale. The program can
    also play a sequence of chords and can
    find a key from a list of chords.
    the most impresive part is it
    can hear a chord/note and predict which cord is it
    My version uses AI but i will upload a version
    without AI untill I get it better.
    for now it can guess simple chords.
    This version works for my guitar but 
    it should also work for any other instrument.
    
    Maayan alkelai.
    321maayan@gmail.com
    """


    """Here is an baisic use of the program."""


    # list of chords. it will guess the key and up pitch it by 2 and return the key again
    chords = ["Am", "Em", "F", "G"]
    print("Key = ", get_key_from_chords(chords))
    print("Key after change of 2 = ", get_key_from_chords(key_pitch_change(chords, 2)))
    
   
    
    
    # this part do a countdown till the start of the recording
    print("3")
    time.sleep(.3)
    print("2")
    time.sleep(.3)
    print("1")
    time.sleep(.3)

    # gets the audio then do the math to find its chord
    print("Getting chord")
    chord = get_chord_from_sound(3)
    print(f"You Played the Chord {chord}!!!\nHere is the sound of it")

    # wait 300 ms then play the chord that the program think you played
    time.sleep(.3)
    playChord(chord,2)
    



    # just play an Am for 1.5 seconds
    playChord("Am", 1.5) # notes: A, C, E


    #gets 4 chords then plays them 4 times
    print("Enter 4 chords one chord at a time:")
    user_chords = get_user_chords(4)
    print(f"The chords are: {user_chords}\nand they're at the key of: {get_key_from_chords(user_chords)}")
    
    chord_sequence = user_chords*4
    durations = [2, 1, 2, 1]*4 # the times for each chord
    playChordSequence(chord_sequence, durations)
    print("Thank you for using my program")



if __name__ == "__main__":
    main()