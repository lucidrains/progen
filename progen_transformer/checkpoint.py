# for handling saving and loading checkpoints
# from either filesystem or google cloud storage
import os, errno

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
