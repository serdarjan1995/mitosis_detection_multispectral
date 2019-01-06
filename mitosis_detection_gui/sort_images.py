import os
import glob

list_all = glob.glob('M04_05_sorted\M04_05/*.bmp')
list_sorted = glob.glob('M04_05_sorted\M04_05/M04_05[a-d]_0[0-9]06.bmp')

for file in list_all:
    if file not in list_sorted:
        os.remove(file)