"""
@author : hogan
time:
function:
"""
import tkinter as tk

def clear_and_redraw():
    # 清除子窗口的所有组件
    for widget in child_window.winfo_children():
        widget.destroy()

    # 布置新组件
    new_label = tk.Label(child_window, text="New Label")
    new_label.pack()

    new_button = tk.Button(child_window, text="New Button", command=clear_and_redraw)
    new_button.pack()

root = tk.Tk()
root.title("Main Window")

def open_child_window():
    global child_window
    child_window = tk.Toplevel(root)
    child_window.title("Child Window")

    label = tk.Label(child_window, text="This is a child window!")
    label.pack()

    button = tk.Button(child_window, text="Clear and Redraw", command=clear_and_redraw)
    button.pack()
    button.pack()

open_button = tk.Button(root, text="Open Child Window", command=open_child_window)
open_button.pack()

root.mainloop()
# tuple_data = (1, 2, 3)
# list_data = list(tuple_data)
# print(list_data)
