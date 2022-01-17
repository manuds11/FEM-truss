
import tkinter as tk                    
from tkinter import ttk
from GUI_functions import *
  
root = tk.Tk()
root.title("FEM_TRUSS")
root.geometry("1500x900")

tabControl = ttk.Notebook(root)


Geometry = tk.Frame(tabControl)
Geometry.grid()
Results = tk.Frame(tabControl)

tabControl.add(Geometry, text ='Geometry and Constraints')
tabControl.add(Results, text ='Results')
tabControl.pack(expand = True, fill ="both")


# ************************************
# *********** Geometry Tab ***********
# ******** ***************** *********

# Node panel

# 0
Nodes_lab = tk.Label(Geometry, text = "Node", justify = "center", font = "bold" )
Nodes_lab.grid(row = 0, column = 0, columnspan = 2, pady = 15)
# 1
Node_x_lab = tk.Label(Geometry, text = "X", justify = "center", )
Node_x_lab.grid(row = 1, column = 0, ) 
Node_y_lab = tk.Label(Geometry, text = "Y", justify = "center", )
Node_y_lab.grid(row = 1, column = 1)
# 2
node_x_inp, node_y_inp = float, float
x_entry = tk.Entry(Geometry, textvariable = node_x_inp, width = 12)
x_entry.grid(row = 2, column = 0, padx = 5, pady = 2)
y_entry = tk.Entry(Geometry, textvariable = node_y_inp, width = 12)
y_entry.grid(row = 2, column = 1, padx = 5, pady = 2)
# 3
restrict_x_lab = tk.Label(Geometry, text = "X restricted?", justify = "center", )
restrict_x_lab.grid(row = 3, column = 0, ) 
restrict_y_lab = tk.Label(Geometry, text = "Y restricted?", justify = "center", )
restrict_y_lab.grid(row = 3, column = 1)
# 4
restrict_x_inp, restrict_y_inp = bool, bool
restrict_x_entry = tk.Entry(Geometry, textvariable = restrict_x_inp, width = 12)
restrict_x_entry.grid(row = 4, column = 0, padx = 5, pady = 2)
restrict_y_entry = tk.Entry(Geometry, textvariable = restrict_y_inp, width = 12)
restrict_y_entry.grid(row = 4, column = 1, padx = 5, pady = 2)
#5

Fx_lab = tk.Label(Geometry, text = "Fx", justify = "center", )
Fx_lab.grid(row = 5, column = 0, ) 
Fy_lab = tk.Label(Geometry, text = "Fy", justify = "center", )
Fy_lab.grid(row = 5, column = 1)
# 
Fx_inp, Fy_inp = bool, bool
Fx_entry = tk.Entry(Geometry, textvariable = Fx_inp, width = 12)
Fx_entry.grid(row = 6, column = 0, padx = 5, pady = 2)
Fy_entry = tk.Entry(Geometry, textvariable = Fy_inp, width = 12)
Fy_entry.grid(row = 6, column = 1, padx = 5, pady = 2)

# 7
Gen_Node_but = tk.Button(Geometry, text = "Add Node", height = 1, width = 15,
            command = lambda: AddNode(
                                    Model_GUI,
                                    node_x_inp, node_y_inp,
                                    restrict_x_inp, restrict_y_inp,
                                    Fx_inp, Fy_inp
                                      ), )
Gen_Node_but.grid(row = 7, column = 0, padx = 10, pady = 2, columnspan = 2)
# 8
No_of_nodes = int
Nodes_no_lab = tk.Label(Geometry, text = "Nº of nodes:", justify = "center", font = "bold")
Nodes_no_lab.grid(row = 8, column = 0, padx = 5, pady = 5)
Nodes_no_entry = tk.Entry(Geometry, textvariable = No_of_nodes, width = 12)
Nodes_no_entry.grid(row = 8, column = 1, padx = 5, pady = 5)





# ************** MAIN FUNCTION *************

Model_GUI = Model()







# END
root.mainloop()



