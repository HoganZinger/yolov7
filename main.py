"""
@author : hogan
time:
function: 主界面，实现不同药材显微结构检测界面的选择
"""
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk
import os
from functions import tools

# 创建选择主界面
window = tk.Tk()
window.title("欢迎使用中药材检测系统")
# 设置界面内容
window.geometry('800x600')
window.resizable(False, False)
# 加载 PNG 图像
os_path = os.path.dirname(__file__)
bg_path = os.path.join(os_path, './background/background.png')
bg_image = Image.open(bg_path)
bg_image = bg_image.resize((800, 600))
bg_photo = ImageTk.PhotoImage(bg_image)
# 创建一个 Canvas
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack(fill="both", expand=True)
# 在 Canvas 上显示背景图像
canvas.create_image(0, 0, anchor="nw", image=bg_photo)
canvas.tag_lower(bg_photo)  # 将图像放置在最底层

micro_button_path = os.path.join(os_path, './button/micro_button_default.png')
trait_button_path = os.path.join(os_path, './button/trait_button_default.png')

micro_button_photo = tools.load_image(micro_button_path)
trait_button_photo = tools.load_image(trait_button_path)

micro_button = Button(window, image=micro_button_photo, width=200, height=50)
trait_button = Button(window, image=trait_button_photo, width=200, height=50)
micro_button.pack(padx=20, pady=10)
trait_button.pack(padx=20, pady=10)

micro_button.place(relx=0.38, rely=0.3)
trait_button.place(relx=0.38, rely=0.48)

window.mainloop()
