"""
@author : hogan
time:
function:
"""
from interface import micro_south, micro_suanzaoren, trait_south, trait_suanzaoren, trait_cheqianzi
import tkinter as tk

# 创建选择主界面
window = tk.Tk()
window.title("欢迎使用中药材显微结构检测系统")
# 设置界面内容
window.geometry('650x450')
window.resizable(False, False)

def open_interface_1():
    micro_south.show_interface_1()

def open_interface_2():
    micro_suanzaoren.show_interface_2()

# 创建按钮1，指向界面1
button1 = tk.Button(window, text="南方菟丝子显微结构检测", command=open_interface_1)
button1.pack()

# 创建按钮2，指向界面2
button2 = tk.Button(window, text="酸枣仁显微结构检测", command=open_interface_2)
button2.pack()

button1.place(relx=0.39, rely=0.25)
button2.place(relx=0.41, rely=0.35)