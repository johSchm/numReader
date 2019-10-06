#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
description:    ---
author:         Johann Schmidt
date:           October 2019
"""

import tkinter as tk
import turtle
import time
import pyscreenshot as ImageGrab
from PIL import Image
import io
import os
import subprocess
import numpy as np
import cv2


class FreeDrawingApp(tk.Frame):
    """ Free Drawing Application
    """

    def __init__(self, parent, pensize=5):
        """ Initialization method.
        :param parent: parent frame
        :param pensize
        """
        tk.Frame.__init__(self, parent.obj())

        self.parent = parent.obj()
        self.pensize = pensize

        self.cv = tk.Canvas(self)
        self.cv.pack()
        self.cv.bind("<B1-Motion>", self.paint)

        tk.Button(self, text="Clear", command=self.clear).pack()
        self.grid(row=0, column=0, sticky='nsew')
        self.rowconfigure(0, weight=10)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        message = tk.Label(self, text="Press and Drag the mouse to draw")
        message.pack(side=tk.BOTTOM)

        self.parent.update()

    def paint(self, event):
        """ Draws a dot (oval) at the current event position.
        :param event
        """
        if self.cv is not None:
            x1, y1 = (event.x - (self.pensize / 2)), (event.y - (self.pensize / 2))
            x2, y2 = (event.x + (self.pensize / 2)), (event.y + (self.pensize / 2))
            self.cv.create_oval(x1, y1, x2, y2, fill="#000000")

    def save(self, filename="img.jpg"):
        """ Saves the snapshot of the canvas.
        :param filename
        """
        ps = self.cv.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save(filename, 'jpeg')

    def clear(self):
        """ Clears the canvas.
        """
        if self.cv is not None:
            self.cv.delete("all")

    def pixel_array(self):
        """ Converts the canvas content as a pixel array.
        """
        self.save("img.jpg")
        img = cv2.imread("img.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        img = cv2.bitwise_not(img)
        cv2.imwrite("img2.jpg", img)
        return img

    def canvas(self):
        """ Returns the canvas coordinates.
        :return: x, y, x1, y1
        """
        if self.cv is None:
            return None
        x = self.cv.winfo_rootx() + self.cv.winfo_x()
        y = self.cv.winfo_rooty() + self.cv.winfo_y()
        x1 = x + self.cv.winfo_width()
        y1 = y + self.cv.winfo_height()
        return x, y, x1, y1

    def drag_handler(self, x, y):
        """ Moves turtle handler to new position.
        :param x:
        :param y:
        """
        if self.turtle_handler is not None:
            self.turtle_handler.ondrag(None)
            self.turtle_handler.goto(x, y)
            self.turtle_handler.ondrag(self.drag_handler)

    def run(self, interrupt_method=None, interrupt_interval=5, **kwargs):
        """ Starts the drawing routine.
        :param interrupt_method
        :param interrupt_interval in s
        """
        i = 0
        while True:
            self.parent.update_idletasks()
            self.parent.update()
            if interrupt_interval >= 0 and i > 0 and i % (interrupt_interval * 100) == 0:
                interrupt_method(*tuple(value for _, value in kwargs.items()))
            time.sleep(0.01)
            i += 1
