import subprocess


print("training on venice")
subprocess.run(['python', './train.py', '-f', 'E:/Alessandro/venice', '-o', 'adam'])
print("training on shanghai B")
subprocess.run(['python', './train.py', '-f', 'E:/Alessandro/ShanghaiTech/part_B_final', '-o', 'adam'])
print("training on shanghai A")
subprocess.run(['python', './train.py', '-f', 'E:/Alessandro/ShanghaiTech/part_A_final', '-o', 'sgd'])
