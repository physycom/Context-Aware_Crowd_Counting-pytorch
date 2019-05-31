import subprocess

subprocess.run(['python', './train.py', '-f', 'D:/Alex/venice', '-o', 'adam'])
subprocess.run(['python', './train.py', '-f', 'D:/Alex/ShanghaiTech/part_B_final', '-o', 'adam'])
subprocess.run(['python', './train.py', '-f', 'D:/Alex/ShanghaiTech/part_A_final', '-o', 'sgd'])
