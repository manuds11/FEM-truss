# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 20:30:22 2022

@author: manud
"""
import GUI_truss, Truss_simple_structure
from tkinter import messagebox



def AddNode(Model, node_x_inp, node_y_inp, restrict_x_inp, restrict_y_inp, Fx_inp, Fy_inp):
    if (type(node_x_inp) != "<class 'float'>"):
        messagebox.showwarning("Error", "Please enter a decimal number as the X coordinate.")
    if (type(node_y_inp) != 'float'):
        messagebox.showwarning("Error", "Please enter a decimal number as the Y coordinate.")
    if type(restrict_x_inp) != 'bool' or type(restrict_y_inp) != 'bool':
        messagebox.showwarning("Error", "You have entered a non boolean entry as DOF restrictions.")
    if (type(Fx_inp) != "float"):
        messagebox.showwarning("Error", "Please enter a decimal number as the Fy applied.")
    if (type(node_y_inp) != 'float'):
        messagebox.showwarning("Error", "Please enter a decimal number as the Fx applied.")
        
    Model.AddNode(Node((node_x_inp(), node_y_inp), (restrict_x_inp, restrict_y_inp), (Fx, Fy)))