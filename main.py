"""
@author : hogan
time:
function: 主界面，实现不同药材显微结构检测界面的选择
"""
import tkinter as tk

import interface_1
import interface_2

def open_interface_1():
    interface_1.show_interface_1()

def open_interface_2():
    interface_2.show_interface_2()


# 创建选择主界面
window = tk.Tk()
window.title("中药材显微结构检测系统")
# 设置界面内容
window.geometry('650x450')

# 创建按钮1，指向界面1
button1 = tk.Button(window, text="南方菟丝子显微结构检测", command=open_interface_1)
button1.pack()

# 创建按钮2，指向界面2
button2 = tk.Button(window, text="酸枣仁显微结构检测", command=open_interface_2)
button2.pack()


window.mainloop()