
import threading
import time

def task1():
    for i in range(5):
        print(f'Task 1: {i}')
        
        time.sleep(0.5)

def task2():
    for i in range(5):
        print(f'Task 2: {i}')
        time.sleep(0.5)

# 建立 Thread 物件
t1 = threading.Thread(target=task1)
t2 = threading.Thread(target=task2)

# 啟動線程
t1.start()
t2.start()

# 等待線程結束
t1.join()
t2.join()
print('All threads finished.')