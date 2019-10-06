import tkinter as tk
from tkinter import messagebox


x = True


def update_x():
    global x
    x = False


class Window:
    """ GUI Window.
    """

    def __init__(self, title="App", width=300, height=300):
        """ Initialization method.
        :param title
        :param width
        :param height
        """
        self.title = title
        self.width = width
        self.height = height
        self.window = tk.Tk()
        self.setup_geometry()
        self.window.resizable(0, 0)
        self.window.title(title)
        self.window.protocol('WM_DELETE_WINDOW', self.confirm_exit)
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)

    def setup_geometry(self):
        """ Sets the geometry of the window.
        """
        if self.window is not None \
                and self.width is not None \
                and self.height is not None:
            position_right = int(self.window.winfo_screenwidth() / 2 - self.width / 2)
            position_down = int(self.window.winfo_screenheight() / 2 - self.height / 2)
            self.window.geometry(
                str(self.height) + 'x' + str(self.width)
                + '+' + str(position_right) + '+' + str(position_down))

    def confirm_exit(self):
        """ Quits the application.
        """
        self.window.destroy()

    def obj(self):
        """ Returns the window object.
        :return object
        """
        return self.window
